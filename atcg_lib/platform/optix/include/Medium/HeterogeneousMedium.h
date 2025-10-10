#pragma once

#include <Core/Memory.h>
#include <Medium/Medium.h>
#include <Medium/HeterogeneousMediumData.cuh>
#include <DataStructure/Dictionary.h>
#include <DataStructure/BoundingBox.h>
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

struct HeterogeneousMediumComponent
{
    HeterogeneousMediumComponent() = default;

    ATCG_INLINE atcg::ref_ptr<Texture3D> density() const
    {
        return AssetManager::getAsset<Texture3D>(density_grid.handle);
    }
    ATCG_INLINE atcg::ref_ptr<Texture3D> albedo() const
    {
        return AssetManager::getAsset<Texture3D>(albedo_grid.handle);
    }
    ATCG_INLINE atcg::ref_ptr<Texture3D> emission() const
    {
        return AssetManager::getAsset<Texture3D>(emission_grid.handle);
    }

    struct GridComponent
    {
        BoundingBox bbox;
        AssetHandle handle = 0;
        float scale        = 1.0f;
    };

    GridComponent density_grid;
    GridComponent albedo_grid;
    GridComponent emission_grid;

    float g = 0.0f;

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Heterogeneous Medium"; }
};

namespace GUI
{
ATCG_DECLARE_COMPONENT_GUI_RENDERER(HeterogeneousMediumComponent);
}
namespace Serialization
{
ATCG_DECLARE_COMPONENT_SERIALIZER(HeterogeneousMediumComponent);
}

}    // namespace atcg