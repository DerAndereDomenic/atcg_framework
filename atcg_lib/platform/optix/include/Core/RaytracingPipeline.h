#pragma once

#include <Core/API.h>
#include <Core/RaytracingContext.h>
#include <Core/TraceParameters.h>
#include <optix.h>

namespace atcg
{

struct ShaderEntryPointDesc
{
    std::string ptx_filename;
    std::string entrypoint_name;
};

class ATCG_API RayTracingPipeline
{
public:
    /**
     * @brief Create a raytracing pipeline
     *
     * @param context The optix context
     */
    RayTracingPipeline(const atcg::ref_ptr<RaytracingContext>& context, const uint32_t num_rays = 1);

    /**
     * @brief Destructor
     */
    ~RayTracingPipeline();

    /**
     * @brief Add a raygen shader
     *
     * @param raygen_shader_desc The entry point description
     *
     * @return The program group associated with the entry function
     */
    OptixProgramGroup addRaygenShader(const ShaderEntryPointDesc& raygen_shader_desc);

    /**
     * @brief Add a callable shader
     *
     * @param callable_shader_desc The entry point description
     *
     * @return The program group associated with the entry function
     */
    OptixProgramGroup addCallableShader(const ShaderEntryPointDesc& callable_shader_desc);

    /**
     * @brief Add a miss shader
     *
     * @param miss_shader_desc The entry point description
     *
     * @return The program group associated with the entry function
     */
    OptixProgramGroup addMissShader(const ShaderEntryPointDesc& miss_shader_desc);

    /**
     * @brief Add a triangle hit shader
     *
     * @param shape_type The shape type
     * @param shader_slot The shader slot
     * @param closestHit_shader_desc The entry point description for the closest hit shader
     * @param anyHit_shader_desc The entry point description for the any hit shader
     *
     * @return The program group associated with the entry function
     */
    OptixProgramGroup addTrianglesHitGroupShader(const std::string& shape_type,
                                                 const uint32_t shader_slot,
                                                 const ShaderEntryPointDesc& closestHit_shader_desc,
                                                 const ShaderEntryPointDesc& anyHit_shader_desc);

    /**
     * @brief Create the pipeline object
     */
    void createPipeline();

    /**
     * @brief Get the pipeline object
     *
     * @return The pipeline
     */
    OptixPipeline getPipeline() const;

    /**
     * @brief Get the number of rays supported by the pipeline
     *
     * @return The number of rays
     */
    uint32_t numRays() const;

    /**
     * @brief Get the ray program groups associated with a ray type
     *
     * @param ray_type The ray type
     *
     * @return The program groups
     */
    const std::vector<OptixProgramGroup>& getRayProgramGroups(const std::string& ray_type) const;

    /**
     * @brief Get the trace parameters for a given ray type index and miss index
     *
     * @param ray_type_index The ray type index
     * @param miss_index The miss index
     * @param occlusion Whether this is an occlusion ray
     *
     * @return The trace parameters
     */
    TraceParameters getRay(const uint32_t ray_type_index, const uint32_t miss_index, bool occlusion = false) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg