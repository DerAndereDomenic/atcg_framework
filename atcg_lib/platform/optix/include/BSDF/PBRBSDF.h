#pragma once

#include <BSDF/BSDF.h>
#include <BSDF/PBRBSDFData.cuh>
#include <Renderer/Material.h>
#include <Core/PipelineInitializer.h>

namespace atcg
{
/**
 * @brief A PBR BSDF
 */
class PBRBSDF : public BSDF
{
public:
    /**
     * @brief Construct a BSDF with arbitrary parameters.
     * Input parameters:
     * - "material": atcg::Material
     *
     * @param dict Dictionary holding the parameters
     */
    PBRBSDF(const atcg::Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~PBRBSDF();

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() {};

    ATCG_INLINE atcg::dref_ptr<PBRBSDFData> getDataBuffer() const { return _bsdf_data_buffer; }

private:
    atcg::ref_ptr<Texture2D> _diffuse_texture;
    atcg::ref_ptr<Texture2D> _metallic_texture;
    atcg::ref_ptr<Texture2D> _roughness_texture;

    atcg::dref_ptr<PBRBSDFData> _bsdf_data_buffer;
};

ATCG_DECLARE_COMPONENT_PIPELINE_INITIALIZER(PBRBSDF);
}    // namespace atcg