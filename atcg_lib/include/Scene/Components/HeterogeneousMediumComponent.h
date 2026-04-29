#pragma once

#include <Asset/AssetManagerSystem.h>
#include <DataStructure/BoundingBox.h>
#include <Renderer/Texture.h>
#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentSerializer.h>
#include <Scene/ComponentRenderer.h>

namespace atcg
{
struct HeterogeneousMediumComponent
{
    HeterogeneousMediumComponent()
    {
        // Create default 1x1x1 textures for density, albedo and emission
        TextureSpecification spec;
        spec.width  = 1;
        spec.height = 1;
        spec.depth  = 1;
        spec.format = TextureFormat::RGBFLOAT;
        glm::vec3 black(0);
        glm::vec3 white(1.0f);
        _default_emission_texture = atcg::Texture3D::create(&black, spec);

        _default_albedo_texture = atcg::Texture3D::create(&white, spec);
    }

    ATCG_INLINE atcg::ref_ptr<Texture3D> density() const
    {
        return AssetManager::getAsset<Texture3D>(density_grid.handle);
    }
    ATCG_INLINE atcg::ref_ptr<Texture3D> albedo() const
    {
        return AssetManager::isAssetHandleValid(albedo_grid.handle)
                   ? AssetManager::getAsset<Texture3D>(albedo_grid.handle)
                   : _default_albedo_texture;
    }
    ATCG_INLINE atcg::ref_ptr<Texture3D> emission() const
    {
        return AssetManager::isAssetHandleValid(emission_grid.handle)
                   ? AssetManager::getAsset<Texture3D>(emission_grid.handle)
                   : _default_emission_texture;
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

private:
    atcg::ref_ptr<Texture3D> _default_emission_texture;
    atcg::ref_ptr<Texture3D> _default_albedo_texture;
};

ATCG_DECLARE_COMPONENT_RENDERER(HeterogeneousMediumComponent);

namespace Serialization
{
ATCG_DECLARE_COMPONENT_SERIALIZER(HeterogeneousMediumComponent);
}

namespace GUI
{
ATCG_DECLARE_COMPONENT_GUI_RENDERER(HeterogeneousMediumComponent);
}

}    // namespace atcg