#pragma once

#include <BSDF/BSDF.h>
#include <BSDF/PBRBSDFData.cuh>
#include <Renderer/Material.h>

namespace atcg
{
class PBRBSDF : public BSDF
{
public:
    PBRBSDF(const Material& material);

    ~PBRBSDF();

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    cudaArray_t _diffuse_texture;
    cudaArray_t _metallic_texture;
    cudaArray_t _roughness_texture;

    atcg::dref_ptr<PBRBSDFData> _bsdf_data_buffer;
};
}    // namespace atcg