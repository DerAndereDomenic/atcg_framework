#pragma once

#include <Pathtracing/OptixInterface.h>
#include <Renderer/Material.h>

namespace atcg
{

class PBRBSDF : public OptixBSDF
{
public:
    PBRBSDF(const Material& material);

    ~PBRBSDF();

    void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                            const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    cudaArray_t _diffuse_texture;
    cudaArray_t _metallic_texture;
    cudaArray_t _roughness_texture;

    atcg::dref_ptr<PBRBSDFData> _bsdf_data_buffer;
};

class RefractiveBSDF : public OptixBSDF
{
public:
    RefractiveBSDF(const Material& material);

    ~RefractiveBSDF();
    void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                            const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    cudaArray_t _diffuse_texture;
    atcg::dref_ptr<RefractiveBSDFData> _bsdf_data_buffer;
};

}    // namespace atcg