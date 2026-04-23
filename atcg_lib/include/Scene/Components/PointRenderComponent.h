#pragma once

#include <Asset/AssetManagerSystem.h>
#include <Core/glm.h>
#include <Renderer/Shader.h>
#include <Renderer/ShaderManager.h>
#include <Scene/Components/RenderComponent.h>

namespace atcg
{
struct PointRenderComponent : public RenderComponent
{
    PointRenderComponent(const glm::vec3& color = glm::vec3(1), const float& point_size = 1.0f)
        : RenderComponent(),
          color(color),
          point_size(point_size)
    {
    }
    PointRenderComponent(const atcg::ref_ptr<Shader>& shader,
                         const glm::vec3& color  = glm::vec3(1),
                         const float& point_size = 1.0f)
        : RenderComponent(),
          color(color),
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

    ATCG_INLINE atcg::ref_ptr<Shader> shader() const
    {
        auto shader = AssetManager::getAsset<Shader>(shader_handle);
        return shader ? shader : default_shader;
    }

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Point Renderer"; }

    atcg::ref_ptr<Shader> default_shader = atcg::ShaderManager::getShader("base");
    glm::vec3 color                      = glm::vec3(1);
    float point_size                     = 1.0f;

    AssetHandle shader_handle = 0;
};
}    // namespace atcg