#pragma once

#include <Core/glm.h>

namespace atcg
{
struct PointLightComponent
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

}    // namespace atcg