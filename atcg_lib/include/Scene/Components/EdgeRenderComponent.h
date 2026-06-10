#pragma once

#include <Scene/Components/RenderComponent.h>
#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentRenderer.h>
#include <Scene/ComponentSerializer.h>

namespace atcg
{
struct ATCG_API EdgeRenderComponent : public RenderComponent
{
    EdgeRenderComponent(const glm::vec3& color = glm::vec3(1)) : RenderComponent(), color(color) {}

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Edge Renderer"; }

    glm::vec3 color = glm::vec3(1);
    // TODO Edge radius?
};

ATCG_DECLARE_COMPONENT_RENDERER(EdgeRenderComponent);

namespace Serialization
{
ATCG_DECLARE_COMPONENT_SERIALIZER(EdgeRenderComponent);
}

namespace GUI
{
ATCG_DECLARE_COMPONENT_GUI_RENDERER(EdgeRenderComponent);
}

}    // namespace atcg