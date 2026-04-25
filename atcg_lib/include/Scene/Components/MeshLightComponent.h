#pragma once

#include <Asset/AssetManagerSystem.h>
#include <Renderer/Texture.h>
#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentRenderer.h>

namespace atcg
{
struct MeshLightComponent
{
    MeshLightComponent()
    {
        glm::vec3 color(1);
        TextureSpecification spec_emissive;
        spec_emissive.width  = 1;
        spec_emissive.height = 1;
        glm::u8vec4 color_quant((uint8_t)(color[0] * 255.0f),
                                (uint8_t)(color[1] * 255.0f),
                                (uint8_t)(color[2] * 255.0f),
                                (uint8_t)(255.0f));
        _emissive_texture = atcg::Texture2D::create(&color_quant, spec_emissive);
    }

    float intensity = 1.0f;

    ATCG_INLINE void setEmissiveColor(const glm::vec3& color)
    {
        TextureSpecification spec_emissive;
        spec_emissive.width  = 1;
        spec_emissive.height = 1;
        glm::u8vec4 color_quant((uint8_t)(color[0] * 255.0f),
                                (uint8_t)(color[1] * 255.0f),
                                (uint8_t)(color[2] * 255.0f),
                                (uint8_t)(255.0f));
        _emissive_texture = atcg::Texture2D::create(&color_quant, spec_emissive);
    }

    ATCG_INLINE atcg::ref_ptr<atcg::Texture2D> getEmissiveTexture() const
    {
        auto texture = AssetManager::getAsset<Texture2D>(emissive_handle);
        return texture ? texture : _emissive_texture;
    }

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Mesh Light"; }

    AssetHandle emissive_handle;

private:
    atcg::ref_ptr<atcg::Texture2D> _emissive_texture;
};

ATCG_DECLARE_COMPONENT_RENDERER(MeshLightComponent);

namespace GUI
{
ATCG_DECLARE_COMPONENT_GUI_RENDERER(MeshLightComponent);
}

}    // namespace atcg