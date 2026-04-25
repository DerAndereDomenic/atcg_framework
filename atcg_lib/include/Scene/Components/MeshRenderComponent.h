#pragma once

#include <Asset/AssetManagerSystem.h>
#include <Renderer/Shader.h>
#include <Renderer/ShaderManager.h>
#include <Scene/Components/RenderComponent.h>
#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentRenderer.h>

namespace atcg
{
struct MeshRenderComponent : public RenderComponent
{
    MeshRenderComponent() : RenderComponent() {}
    MeshRenderComponent(const atcg::ref_ptr<Shader>& shader) : RenderComponent()
    {
        if(AssetManager::isAssetHandleValid(shader->handle))
        {
            shader_handle = shader->handle;
        }
        else
        {
            shader_handle = AssetManager::registerAsset(shader, "shader");
        }
    }

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

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Mesh Renderer"; }

    atcg::ref_ptr<Shader> default_shader = atcg::ShaderManager::getShader("base");
    bool receive_shadow                  = true;

    AssetHandle material_handle = 0;
    AssetHandle shader_handle   = 0;
};

ATCG_DECLARE_COMPONENT_RENDERER(MeshRenderComponent);

namespace GUI
{
ATCG_DECLARE_COMPONENT_GUI_RENDERER(MeshRenderComponent);
}
}    // namespace atcg