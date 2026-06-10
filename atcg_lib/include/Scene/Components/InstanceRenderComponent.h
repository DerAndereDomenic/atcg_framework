#pragma once

#include <Asset/AssetManagerSystem.h>
#include <Scene/Components/RenderComponent.h>
#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentRenderer.h>
#include <Scene/ComponentSerializer.h>

namespace atcg
{
struct ATCG_API InstanceRenderComponent : public RenderComponent
{
    InstanceRenderComponent() : RenderComponent() {}

    void addInstanceBuffer(const atcg::ref_ptr<VertexBuffer>& instance_vbo) { instance_vbos.push_back(instance_vbo); }

    ATCG_INLINE atcg::ref_ptr<Material> material() const
    {
        auto mat = AssetManager::getAsset<Material>(material_handle);
        return mat ? mat : default_material;
    }

    ATCG_INLINE atcg::ref_ptr<Shader> shader() const
    {
        auto shader = AssetManager::getAsset<Shader>(shader_handle);
        return shader ? shader : default_shader;
    }

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Instance Renderer"; }

    std::vector<atcg::ref_ptr<VertexBuffer>> instance_vbos;
    atcg::ref_ptr<atcg::Shader> default_shader = nullptr;
    bool receive_shadow                        = true;

    AssetHandle material_handle = 0;
    AssetHandle shader_handle;
};

ATCG_DECLARE_COMPONENT_RENDERER(InstanceRenderComponent);

namespace Serialization
{
ATCG_DECLARE_COMPONENT_SERIALIZER(InstanceRenderComponent);
}


namespace GUI
{
ATCG_DECLARE_COMPONENT_GUI_RENDERER(InstanceRenderComponent);
}

}    // namespace atcg