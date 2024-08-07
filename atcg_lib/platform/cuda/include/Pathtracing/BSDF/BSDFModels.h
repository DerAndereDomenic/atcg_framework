#pragma once

#include <Pathtracing/OptixInterface.h>
#include <Renderer/Material.h>

namespace atcg
{

/**
 * @brief A pbr bsdf
 */
class PBRBSDF : public OptixBSDF
{
public:
    /**
     * @brief Create the bsdf
     *
     * @param material The material that describes the bsdf
     */
    PBRBSDF(const Material& material);

    /**
     * @brief Destructor
     */
    ~PBRBSDF();

    /**
     * @brief Initialize the optix pipeline.
     * Each Optix component has to initialize its part of the raytracing pipeline by defining appropriate entry points
     * and sbt entries.
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                            const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    cudaArray_t _diffuse_texture;
    cudaArray_t _metallic_texture;
    cudaArray_t _roughness_texture;

    atcg::dref_ptr<PBRBSDFData> _bsdf_data_buffer;
};

/**
 * @brief Class to model a refractive (dielectric) bsdf
 */
class RefractiveBSDF : public OptixBSDF
{
public:
    /**
     * @brief Create the bsdf
     *
     * @param material The material that describes the bsdf
     */
    RefractiveBSDF(const Material& material);

    /**
     * @brief Destructor
     */
    ~RefractiveBSDF();

    /**
     * @brief Initialize the optix pipeline.
     * Each Optix component has to initialize its part of the raytracing pipeline by defining appropriate entry points
     * and sbt entries.
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                            const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    cudaArray_t _diffuse_texture;
    atcg::dref_ptr<RefractiveBSDFData> _bsdf_data_buffer;
};

}    // namespace atcg