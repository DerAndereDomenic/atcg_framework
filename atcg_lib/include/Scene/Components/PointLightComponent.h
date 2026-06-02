#pragma once

#include <Core/glm.h>
#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentRenderer.h>
#include <Scene/ComponentSerializer.h>

namespace atcg
{
struct ATCG_API PointLightComponent
{
    PointLightComponent(const float intensity = 1.0f, const glm::vec3& color = glm::vec3(1))
        : intensity(intensity),
          color(color)
    {
    }

    float intensity  = 1.0f;
    glm::vec3 color  = glm::vec3(1);
    bool cast_shadow = true;

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Point Light"; }
};

ATCG_DECLARE_COMPONENT_RENDERER(PointLightComponent);

namespace Serialization
{
ATCG_DECLARE_COMPONENT_SERIALIZER(PointLightComponent);
}

namespace GUI
{
ATCG_DECLARE_COMPONENT_GUI_RENDERER(PointLightComponent);
}

}    // namespace atcg