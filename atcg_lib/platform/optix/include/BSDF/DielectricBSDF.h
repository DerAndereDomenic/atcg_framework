#pragma once

#include <BSDF/BSDF.h>
#include <BSDF/DielectricBSDFData.cuh>
#include <Renderer/Texture.h>
#include <Core/PipelineInitializer.h>

namespace atcg
{
class DielectricBSDF : public BSDF
{
public:
    /**
     * @brief Construct a BSDF with arbitrary parameters.
     * Input parameters:
     * - "material": atcg::Material
     *
     * @param dict Dictionary holding the parameters
     */
    DielectricBSDF(const atcg::Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~DielectricBSDF();

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override {}

    ATCG_INLINE atcg::dref_ptr<DielectricBSDFData> getDataBuffer() const { return _bsdf_data_buffer; }

private:
    atcg::ref_ptr<Texture2D> _diffuse_texture;
    atcg::ref_ptr<Texture2D> _roughness_texture;
    atcg::ref_ptr<Texture2D> _ior_texture;

    atcg::dref_ptr<DielectricBSDFData> _bsdf_data_buffer;
};

ATCG_DECLARE_COMPONENT_PIPELINE_INITIALIZER(DielectricBSDF);
}    // namespace atcg