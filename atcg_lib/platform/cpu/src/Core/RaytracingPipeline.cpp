#include <Core/RaytracingPipeline.h>

#include <Core/Optix.h>

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

    Impl(const atcg::ref_ptr<RaytracingContext>& context, const uint32_t num_rays);

    ~Impl();
};

RayTracingPipeline::Impl::Impl(const atcg::ref_ptr<RaytracingContext>& context, const uint32_t num_rays) {}

RayTracingPipeline::Impl::~Impl() {}

RayTracingPipeline::RayTracingPipeline(const atcg::ref_ptr<RaytracingContext>& context, const uint32_t num_rays)
{
    impl = std::make_unique<Impl>(context, num_rays);
}

RayTracingPipeline::~RayTracingPipeline() {}

OptixProgramGroup RayTracingPipeline::addRaygenShader(const ShaderEntryPointDesc& raygen_shader_desc)
{
    return nullptr;
}

OptixProgramGroup RayTracingPipeline::addCallableShader(const ShaderEntryPointDesc& callable_shader_desc)
{
    return nullptr;
}

OptixProgramGroup RayTracingPipeline::addMissShader(const ShaderEntryPointDesc& miss_shader_desc)
{
    return nullptr;
}

OptixProgramGroup RayTracingPipeline::addTrianglesHitGroupShader(const std::string& shape_type,
                                                                 const uint32_t shader_slot,
                                                                 const ShaderEntryPointDesc& closestHit_shader_desc,
                                                                 const ShaderEntryPointDesc& anyHit_shader_desc)
{
    return nullptr;
}

void RayTracingPipeline::createPipeline() {}

OptixPipeline RayTracingPipeline::getPipeline() const
{
    return nullptr;
}

uint32_t RayTracingPipeline::numRays() const
{
    return 0;
}

const std::vector<OptixProgramGroup>& RayTracingPipeline::getRayProgramGroups(const std::string& shape_type) const
{
    static std::vector<OptixProgramGroup> empty_vector;
    return empty_vector;
}

TraceParameters
RayTracingPipeline::getRay(const uint32_t ray_type_index, const uint32_t miss_index, bool occlusion) const
{
    TraceParameters trace_params;

    return trace_params;
}

void RayTracingPipeline::launch(CUdeviceptr params,
                                size_t params_size,
                                const OptixShaderBindingTable* sbt,
                                size_t width,
                                size_t height,
                                size_t depth,
                                Stream stream)
{
}

}    // namespace atcg