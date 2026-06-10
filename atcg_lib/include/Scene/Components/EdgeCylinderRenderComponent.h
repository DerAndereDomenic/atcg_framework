#pragma once

#include <Asset/AssetManagerSystem.h>
#include <Renderer/Material.h>
#include <Scene/Components/RenderComponent.h>
#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentRenderer.h>
#include <Scene/ComponentSerializer.h>

namespace atcg
{
struct ATCG_API EdgeCylinderRenderComponent : public RenderComponent
{
    EdgeCylinderRenderComponent(float radius = 0.001f) : RenderComponent(), radius(radius) {}

    ATCG_INLINE atcg::ref_ptr<Material> material() const
    {
        auto mat = AssetManager::getAsset<Material>(material_handle);
        return mat ? mat : default_material;
    }

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Edge Cylinder Renderer"; }

    float radius = 0.001f;

    AssetHandle material_handle = 0;
};

ATCG_DECLARE_COMPONENT_RENDERER(EdgeCylinderRenderComponent);

namespace Serialization
{
ATCG_DECLARE_COMPONENT_SERIALIZER(EdgeCylinderRenderComponent);
}

namespace GUI
{
ATCG_DECLARE_COMPONENT_GUI_RENDERER(EdgeCylinderRenderComponent);
}

}    // namespace atcg