#include <Core/RaytracingPipeline.h>

#include <Core/Common.h>

#include <unordered_map>

#include <optix_stubs.h>

struct OptixProgramGroupEntryKey
{
    OptixModule module = nullptr;
    std::string entryname;

    OptixProgramGroupEntryKey() = default;
    OptixProgramGroupEntryKey(OptixModule _module, const char* _entryname)
        : module {_module},
          entryname {_entryname != nullptr ? _entryname : ""}
    {
    }

    bool operator==(const OptixProgramGroupEntryKey& other) const
    {
        return module == other.module && entryname == other.entryname;
    }
};

struct OptixProgramGroupDescKey
{
    OptixProgramGroupKind kind;
    unsigned int flags = 0;

    std::array<OptixProgramGroupEntryKey, 3> entryPoints;

    OptixProgramGroupDescKey() = default;
    OptixProgramGroupDescKey(const OptixProgramGroupDesc& desc) : kind {desc.kind}, flags {desc.flags}
    {
        switch(kind)
        {
            case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
                entryPoints[0] = {desc.raygen.module, desc.raygen.entryFunctionName};
                break;
            case OPTIX_PROGRAM_GROUP_KIND_MISS:
                entryPoints[0] = {desc.miss.module, desc.miss.entryFunctionName};
                break;
            case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
                entryPoints[0] = {desc.exception.module, desc.exception.entryFunctionName};
                break;
            case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
                entryPoints[0] = {desc.hitgroup.moduleCH, desc.hitgroup.entryFunctionNameCH};
                entryPoints[1] = {desc.hitgroup.moduleAH, desc.hitgroup.entryFunctionNameAH};
                entryPoints[2] = {desc.hitgroup.moduleIS, desc.hitgroup.entryFunctionNameIS};
                break;
            case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
                entryPoints[0] = {desc.callables.moduleDC, desc.callables.entryFunctionNameDC};
                entryPoints[1] = {desc.callables.moduleCC, desc.callables.entryFunctionNameCC};
                break;
        }
    }

    bool operator==(const OptixProgramGroupDescKey& other) const
    {
        return kind == other.kind && flags == other.flags && entryPoints[0] == other.entryPoints[0] &&
               entryPoints[1] == other.entryPoints[1] && entryPoints[2] == other.entryPoints[2];
    }
};


template<typename T>
inline void hash_combine(std::size_t& s, const T& v)
{
    std::hash<T> h;
    s ^= h(v) + 0x9e3779b9 + (s << 6) + (s >> 2);
}

namespace std
{

template<>
struct hash<OptixProgramGroupEntryKey>
{
    size_t operator()(const OptixProgramGroupEntryKey& v) const
    {
        size_t s = 0;
        hash_combine(s, v.module);
        hash_combine(s, v.entryname);
        return s;
    }
};

template<>
struct hash<OptixProgramGroupDescKey>
{
    size_t operator()(const OptixProgramGroupDescKey& v) const
    {
        size_t s = 0;
        hash_combine(s, v.kind);
        hash_combine(s, v.flags);
        for(const auto& e: v.entryPoints)
        {
            hash_combine(s, e);
        }
        return s;
    }
};

}    // end namespace std

namespace atcg
{
class RayTracingPipeline::Impl
{
public:
    Impl() = default;

    Impl(OptixDeviceContext context);

    ~Impl();

    OptixDeviceContext context;
    OptixModuleCompileOptions module_compile_options     = {};
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipelineLinkOptions pipeline_link_options       = {};
    OptixProgramGroupOptions program_group_options       = {};

    std::vector<char> readFile(const char* filename);
    OptixModule createNewModule(const std::string& ptx_filename);
    OptixModule getCachedModule(const std::string& ptx_filename);
    OptixProgramGroup createNewProgramGroup(const OptixProgramGroupDesc& prog_group_desc);
    OptixProgramGroup getCachedProgramGroup(const OptixProgramGroupDesc& prog_group_desc);

    std::unordered_map<std::string, OptixModule> module_cache;
    std::unordered_map<OptixProgramGroupDescKey, OptixProgramGroup> program_groups;

    OptixPipeline pipeline;
    bool pipeline_created = false;
};

RayTracingPipeline::Impl::Impl(OptixDeviceContext context)
{
    this->context = context;

    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    pipeline_compile_options.usesMotionBlur                   = false;
    pipeline_compile_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipeline_compile_options.numPayloadValues                 = 3;
    pipeline_compile_options.numAttributeValues               = 2;
    pipeline_compile_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    pipeline_link_options.maxTraceDepth = 1;
#if OPTIX_VERSION < 70700
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#endif
}

RayTracingPipeline::Impl::~Impl()
{
    if(pipeline_created) optixPipelineDestroy(pipeline);

    for(const auto& [key, prog_group]: program_groups)
    {
        optixProgramGroupDestroy(prog_group);
    }

    for(const auto& [filename, module]: module_cache)
    {
        optixModuleDestroy(module);
    }
}

std::vector<char> RayTracingPipeline::Impl::readFile(const char* filename)
{
    std::ifstream input(filename, std::ios::binary);

    if(!input.is_open())
    {
        throw std::runtime_error("readFile: Could not open file! " + std::string(filename));
    }

    input.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(input.tellg());
    input.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    input.read(buffer.data(), buffer.size());

    return buffer;
}

OptixModule RayTracingPipeline::Impl::createNewModule(const std::string& ptx_filename)
{
    char log[2048];
    size_t sizeof_log = sizeof(log);

    if(ptx_filename.empty())
    {
        return nullptr;
    }

    OptixModule ptx_module;

#if OPTIX_VERSION < 70700
    #define optixModuleCreate optixModuleCreateFromPTX
#endif

    std::vector<char> ptxSource = readFile(ptx_filename.c_str());
    OPTIX_CHECK(optixModuleCreate(context,
                                  &module_compile_options,
                                  &pipeline_compile_options,
                                  ptxSource.data(),
                                  ptxSource.size(),
                                  log,
                                  &sizeof_log,
                                  &ptx_module));
    if(sizeof_log > 1) ATCG_TRACE(log);

#if OPTIX_VERSION < 70700
    #undef optixModuleCreate
#endif

    return ptx_module;
}

OptixModule RayTracingPipeline::Impl::getCachedModule(const std::string& ptx_filename)
{
    OptixModule& ptx_module = module_cache[ptx_filename];
    if(ptx_module == nullptr)
    {
        ptx_module = createNewModule(ptx_filename);
    }
    return ptx_module;
}

OptixProgramGroup RayTracingPipeline::Impl::createNewProgramGroup(const OptixProgramGroupDesc& prog_group_desc)
{
    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixProgramGroup prog_group;
    OPTIX_CHECK(optixProgramGroupCreate(context,
                                        &prog_group_desc,
                                        1,    // num program groups
                                        &program_group_options,
                                        log,
                                        &sizeof_log,
                                        &prog_group));
    if(sizeof_log > 1) ATCG_TRACE(log);
    return prog_group;
}

OptixProgramGroup RayTracingPipeline::Impl::getCachedProgramGroup(const OptixProgramGroupDesc& prog_group_desc)
{
    OptixProgramGroup& prog_group = program_groups[prog_group_desc];
    if(prog_group == nullptr)
    {
        prog_group = createNewProgramGroup(prog_group_desc);
    }
    return prog_group;
}

RayTracingPipeline::RayTracingPipeline(OptixDeviceContext context)
{
    impl = std::make_unique<Impl>(context);
}

RayTracingPipeline::~RayTracingPipeline() {}

OptixProgramGroup RayTracingPipeline::addRaygenShader(const ShaderEntryPointDesc& raygen_shader_desc)
{
    OptixModule ptx_module = impl->getCachedModule(raygen_shader_desc.ptx_filename);

    OptixProgramGroupDesc prog_group_desc    = {};
    prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    prog_group_desc.raygen.module            = ptx_module;
    prog_group_desc.raygen.entryFunctionName = raygen_shader_desc.entrypoint_name.c_str();

    return impl->getCachedProgramGroup(prog_group_desc);
}

OptixProgramGroup RayTracingPipeline::addCallableShader(const ShaderEntryPointDesc& callable_shader_desc)
{
    OptixModule ptx_module = impl->getCachedModule(callable_shader_desc.ptx_filename);

    // TODO directs VS continuation callable!!!

    OptixProgramGroupDesc prog_group_desc = {};
    prog_group_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;

    if(callable_shader_desc.entrypoint_name.rfind("__direct_callable__", 0) == 0)
    {
        prog_group_desc.callables.moduleDC            = ptx_module;
        prog_group_desc.callables.entryFunctionNameDC = callable_shader_desc.entrypoint_name.c_str();
    }
    else if(callable_shader_desc.entrypoint_name.rfind("__continuation_callable__", 0) == 0)
    {
        prog_group_desc.callables.moduleCC            = ptx_module;
        prog_group_desc.callables.entryFunctionNameCC = callable_shader_desc.entrypoint_name.c_str();
    }
    else
    {
        std::stringstream ss;
        ss << "Not a callable program entry point: " << callable_shader_desc.entrypoint_name;
        std::string ss_string = ss.str();
        ATCG_ERROR(ss_string);
        throw std::runtime_error(ss_string);
    }

    return impl->getCachedProgramGroup(prog_group_desc);
}

OptixProgramGroup RayTracingPipeline::addMissShader(const ShaderEntryPointDesc& miss_shader_desc)
{
    OptixModule ptx_module = impl->getCachedModule(miss_shader_desc.ptx_filename);

    OptixProgramGroupDesc prog_group_desc  = {};
    prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    prog_group_desc.miss.module            = ptx_module;
    prog_group_desc.miss.entryFunctionName = miss_shader_desc.entrypoint_name.c_str();

    return impl->getCachedProgramGroup(prog_group_desc);
}

OptixProgramGroup RayTracingPipeline::addTrianglesHitGroupShader(const ShaderEntryPointDesc& closestHit_shader_desc,
                                                                 const ShaderEntryPointDesc& anyHit_shader_desc)
{
    OptixModule ptx_module_ch = impl->getCachedModule(closestHit_shader_desc.ptx_filename);
    OptixModule ptx_module_ah = impl->getCachedModule(anyHit_shader_desc.ptx_filename);

    OptixProgramGroupDesc prog_group_desc = {};
    prog_group_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    prog_group_desc.hitgroup.moduleCH     = ptx_module_ch;
    prog_group_desc.hitgroup.entryFunctionNameCH =
        ptx_module_ch ? closestHit_shader_desc.entrypoint_name.c_str() : nullptr;
    prog_group_desc.hitgroup.moduleAH            = ptx_module_ah;
    prog_group_desc.hitgroup.entryFunctionNameAH = ptx_module_ah ? anyHit_shader_desc.entrypoint_name.c_str() : nullptr;

    return impl->getCachedProgramGroup(prog_group_desc);
}

void RayTracingPipeline::createPipeline()
{
    char log[2048];
    size_t sizeof_log = sizeof(log);

    // Collect all program groups in plain vector
    std::vector<OptixProgramGroup> program_groups;
    std::transform(impl->program_groups.begin(),
                   impl->program_groups.end(),
                   std::back_inserter(program_groups),
                   [](const auto& entry) { return entry.second; });

    OPTIX_CHECK(optixPipelineCreate(impl->context,
                                    &impl->pipeline_compile_options,
                                    &impl->pipeline_link_options,
                                    program_groups.data(),
                                    program_groups.size(),
                                    log,
                                    &sizeof_log,
                                    &impl->pipeline));
    if(sizeof_log > 1) ATCG_TRACE(log);

    impl->pipeline_created = true;
}

OptixPipeline RayTracingPipeline::getPipeline() const
{
    return impl->pipeline;
}

}    // namespace atcg