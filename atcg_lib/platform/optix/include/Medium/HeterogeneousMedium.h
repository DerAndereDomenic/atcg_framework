#pragma once

#include <Core/Memory.h>
#include <Medium/Medium.h>
#include <Medium/HeterogeneousMediumData.cuh>
#include <DataStructure/Dictionary.h>
#include <Renderer/Texture.h>
#include <Asset/AssetManagerSystem.h>

#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentSerializer.h>
#include <Core/PipelineInitializer.h>

namespace atcg
{
class HeterogeneousMedium : public Medium
{
public:
    HeterogeneousMedium(const Dictionary& dict);

    virtual ~HeterogeneousMedium();
    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override {}

    ATCG_INLINE atcg::dref_ptr<HeterogeneousMediumData> getDataBuffer() const { return _data_buffer; }

private:
    atcg::ref_ptr<Texture3D> _density_texture;
    atcg::ref_ptr<Texture3D> _albedo_texture;
    atcg::ref_ptr<Texture3D> _emission_texture;

    atcg::dref_ptr<HeterogeneousMediumData> _data_buffer;
};

ATCG_DECLARE_COMPONENT_PIPELINE_INITIALIZER(HeterogeneousMedium);

}    // namespace atcg