#pragma once

#include <Renderer/BSDFModels.cuh>
#include <Renderer/ShaderBindingTable.h>
#include <Renderer/RaytracingPipeline.h>
#include <Renderer/Material.h>

namespace atcg
{
class BSDF
{
public:
    BSDF() = default;

    virtual ~BSDF() = default;

    inline const BSDFVPtrTable* getBSDFVPtrTable() const { return _vptr_table.get(); }

    virtual void initializeBSDF(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

protected:
    atcg::dref_ptr<BSDFVPtrTable> _vptr_table;
};

class PBRBSDF : public BSDF
{
public:
    PBRBSDF(const Material& material);

    ~PBRBSDF();

    void initializeBSDF(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                        const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    cudaArray_t _diffuse_texture;
    cudaArray_t _metallic_texture;
    cudaArray_t _roughness_texture;

    atcg::dref_ptr<PBRBSDFData> _bsdf_data_buffer;
};

class RefractiveBSDF : public BSDF
{
public:
    RefractiveBSDF(const Material& material);

    ~RefractiveBSDF();
    void initializeBSDF(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                        const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    cudaArray_t _diffuse_texture;
    atcg::dref_ptr<RefractiveBSDFData> _bsdf_data_buffer;
};

}    // namespace atcg