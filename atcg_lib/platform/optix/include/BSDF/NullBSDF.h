#pragma once

#include <BSDF/BSDF.h>
#include <Renderer/Material.h>
#include <Core/PipelineInitializer.h>

namespace atcg
{
/**
 * @brief A PBR BSDF
 */
class NullBSDF : public BSDF
{
public:
    /**
     * @brief Construct a Null BSDF
     *
     * @param dict Dictionary holding the parameters (unused)
     */
    NullBSDF(const atcg::Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~NullBSDF();

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override {}

private:
};

ATCG_DECLARE_COMPONENT_PIPELINE_INITIALIZER(NullBSDF);

}    // namespace atcg