#pragma once

#include <Pathtracing/OptixInterface.h>
#include <Pathtracing/RaytracingShader.h>

namespace atcg
{
/**
 * @brief Interface for optix shader
 */
class OptixRaytracingShader : public OptixComponent, public RaytracingShader
{
public:
    /**
     * @brief Create a shader
     *
     * @param context The optix contexxt
     */
    OptixRaytracingShader(OptixDeviceContext context) : _context(context) {}

    /**
     * @brief Destructor
     */
    virtual ~OptixRaytracingShader() {}

    /**
     * @brief Reset the internal buffers of the shader
     */
    virtual void reset() = 0;

    /**
     * @brief Generate the rays (trace the image)
     *
     * @param pipeline The pipeline
     * @param sbt The sbt
     * @param output The output tensor
     */
    virtual void generateRays(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                              const atcg::ref_ptr<ShaderBindingTable>& sbt,
                              torch::Tensor& output) = 0;

protected:
    OptixDeviceContext _context;
};
}    // namespace atcg