#pragma once

#include <optix.h>

namespace atcg
{

struct ShaderEntryPointDesc
{
    std::string ptx_filename;
    std::string entrypoint_name;
};

class RayTracingPipeline
{
public:
    /**
     * @brief Create a raytracing pipeline
     *
     * @param context The optix context
     */
    RayTracingPipeline(OptixDeviceContext context);

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
     * @param closestHit_shader_desc The entry point description for the closest hit shader
     * @param anyHit_shader_desc The entry point description for the any hit shader
     *
     * @return The program group associated with the entry function
     */
    OptixProgramGroup addTrianglesHitGroupShader(const ShaderEntryPointDesc& closestHit_shader_desc,
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

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg