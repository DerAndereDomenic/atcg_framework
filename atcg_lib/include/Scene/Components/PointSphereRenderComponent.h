#pragma once

#include <Asset/AssetManagerSystem.h>
#include <Core/glm.h>
#include <Renderer/Shader.h>
#include <Renderer/ShaderManager.h>
#include <Scene/Components/RenderComponent.h>

namespace atcg
{
struct PointSphereRenderComponent : public RenderComponent
{
    PointSphereRenderComponent(const float& point_size = 0.1f) : RenderComponent(), point_size(point_size) {}

    PointSphereRenderComponent(const atcg::ref_ptr<Shader>& shader, const float& point_size = 0.1f)
        : RenderComponent(),
          point_size(point_size)
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

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Point Sphere Renderer"; }

    atcg::ref_ptr<Shader> default_shader = atcg::ShaderManager::getShader("base");
    float point_size                     = 0.1f;

    AssetHandle material_handle = 0;
    AssetHandle shader_handle   = 0;
};
}    // namespace atcg