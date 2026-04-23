#pragma once

#include <Scene/Components/RenderComponent.h>

namespace atcg
{
struct EdgeRenderComponent : public RenderComponent
{
    EdgeRenderComponent(const glm::vec3& color = glm::vec3(1)) : RenderComponent(), color(color) {}

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Edge Renderer"; }

    glm::vec3 color = glm::vec3(1);
    // TODO Edge radius?
};
}    // namespace atcg