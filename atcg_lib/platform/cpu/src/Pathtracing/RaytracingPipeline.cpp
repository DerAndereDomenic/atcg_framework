#include <Pathtracing/RaytracingPipeline.h>

#include <unordered_map>


// struct OptixProgramGroupEntryKey
// {
//     OptixModule module = nullptr;
//     std::string entryname;

//     OptixProgramGroupEntryKey() = default;
//     OptixProgramGroupEntryKey(OptixModule _module, const char* _entryname)
//         : module {_module},
//           entryname {_entryname != nullptr ? _entryname : ""}
//     {
//     }

//     bool operator==(const OptixProgramGroupEntryKey& other) const
//     {
//         return module == other.module && entryname == other.entryname;
//     }
// };

// struct OptixProgramGroupDescKey
// {
//     OptixProgramGroupKind kind;
//     unsigned int flags = 0;

//     std::array<OptixProgramGroupEntryKey, 3> entryPoints;

//     OptixProgramGroupDescKey() = default;
//     OptixProgramGroupDescKey(const OptixProgramGroupDesc& desc) : kind {desc.kind}, flags {desc.flags}
//     {
//         switch(kind)
//         {
//             case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
//                 entryPoints[0] = {desc.raygen.module, desc.raygen.entryFunctionName};
//                 break;
//             case OPTIX_PROGRAM_GROUP_KIND_MISS:
//                 entryPoints[0] = {desc.miss.module, desc.miss.entryFunctionName};
//                 break;
//             case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
//                 entryPoints[0] = {desc.exception.module, desc.exception.entryFunctionName};
//                 break;
//             case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
//                 entryPoints[0] = {desc.hitgroup.moduleCH, desc.hitgroup.entryFunctionNameCH};
//                 entryPoints[1] = {desc.hitgroup.moduleAH, desc.hitgroup.entryFunctionNameAH};
//                 entryPoints[2] = {desc.hitgroup.moduleIS, desc.hitgroup.entryFunctionNameIS};
//                 break;
//             case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
//                 entryPoints[0] = {desc.callables.moduleDC, desc.callables.entryFunctionNameDC};
//                 entryPoints[1] = {desc.callables.moduleCC, desc.callables.entryFunctionNameCC};
//                 break;
//         }
//     }

//     bool operator==(const OptixProgramGroupDescKey& other) const
//     {
//         return kind == other.kind && flags == other.flags && entryPoints[0] == other.entryPoints[0] &&
//                entryPoints[1] == other.entryPoints[1] && entryPoints[2] == other.entryPoints[2];
//     }
// };


// template<typename T>
// inline void hash_combine(std::size_t& s, const T& v)
// {
//     std::hash<T> h;
//     s ^= h(v) + 0x9e3779b9 + (s << 6) + (s >> 2);
// }

// namespace std
// {

// template<>
// struct hash<OptixProgramGroupEntryKey>
// {
//     size_t operator()(const OptixProgramGroupEntryKey& v) const
//     {
//         size_t s = 0;
//         hash_combine(s, v.module);
//         hash_combine(s, v.entryname);
//         return s;
//     }
// };

// template<>
// struct hash<OptixProgramGroupDescKey>
// {
//     size_t operator()(const OptixProgramGroupDescKey& v) const
//     {
//         size_t s = 0;
//         hash_combine(s, v.kind);
//         hash_combine(s, v.flags);
//         for(const auto& e: v.entryPoints)
//         {
//             hash_combine(s, e);
//         }
//         return s;
//     }
// };

// }    // end namespace std

namespace atcg
{
class RayTracingPipeline::Impl
{
public:
    Impl() = default;

    Impl(ATCGDeviceContext context);

    ~Impl();

    ATCGModule getCachedModule(const std::string& filename);
    ATCGProgramGroup getCachedProgramGroup(const std::string& name, const ATCGModule& module);

    std::unordered_map<std::string, atcg::ref_ptr<ATCGModule_t>> module_cache;
    std::unordered_map<std::string, atcg::ref_ptr<ATCGProgramGroup_t>> program_cache;
};

RayTracingPipeline::Impl::Impl(ATCGDeviceContext context) {}

RayTracingPipeline::Impl::~Impl() {}

ATCGModule RayTracingPipeline::Impl::getCachedModule(const std::string& filename)
{
    auto& ptx_module = module_cache[filename];
    if(ptx_module == nullptr)
    {
        ptx_module = std::make_shared<ATCGModule_t>(filename.c_str());
    }
    return ptx_module.get();
}

ATCGProgramGroup RayTracingPipeline::Impl::getCachedProgramGroup(const std::string& name, const ATCGModule& module)
{
    auto& group = program_cache[name];
    if(group == nullptr)
    {
        group           = std::make_shared<ATCGProgramGroup_t>();
        group->module   = module;
        group->function = std::make_pair(name, (void*)GetProcAddress(module->handle(), name.c_str()));
    }
    return group.get();
}

RayTracingPipeline::RayTracingPipeline(ATCGDeviceContext context)
{
    impl = std::make_unique<Impl>(context);
}

RayTracingPipeline::~RayTracingPipeline() {}

ATCGProgramGroup RayTracingPipeline::addRaygenShader(const ShaderEntryPointDesc& raygen_shader_desc)
{
    auto module = impl->getCachedModule(raygen_shader_desc.ptx_filename);
    return impl->getCachedProgramGroup(raygen_shader_desc.entrypoint_name, module);
}

ATCGProgramGroup RayTracingPipeline::addCallableShader(const ShaderEntryPointDesc& callable_shader_desc)
{
    auto module = impl->getCachedModule(callable_shader_desc.ptx_filename);
    return impl->getCachedProgramGroup(callable_shader_desc.entrypoint_name, module);
}

ATCGProgramGroup RayTracingPipeline::addMissShader(const ShaderEntryPointDesc& miss_shader_desc)
{
    auto module = impl->getCachedModule(miss_shader_desc.ptx_filename);
    return impl->getCachedProgramGroup(miss_shader_desc.entrypoint_name, module);
}

ATCGProgramGroup RayTracingPipeline::addTrianglesHitGroupShader(const ShaderEntryPointDesc& closestHit_shader_desc,
                                                                const ShaderEntryPointDesc& anyHit_shader_desc)
{
    auto module = impl->getCachedModule(closestHit_shader_desc.ptx_filename);
    return impl->getCachedProgramGroup(closestHit_shader_desc.entrypoint_name, module);
}

void RayTracingPipeline::createPipeline() {}

ATCGPipeline RayTracingPipeline::getPipeline() const
{
    return nullptr;
}

}    // namespace atcg