#pragma once

#include <Core/glm.h>

#include <Material/MediumVPtrTable.h>

namespace atcg
{
class Ray
{
public:
    Ray() = default;

    ATCG_HOST_DEVICE Ray(const glm::vec3& origin, const glm::vec3& direction, float tmin = 0.001f, float tmax = 1e16f)
        : origin(origin),
          direction(direction),
          tmin(tmin),
          tmax(tmax)
    {
    }

    ATCG_HOST_DEVICE ATCG_INLINE glm::vec3 operator()(float t) const { return origin + t * direction; }

    glm::vec3 origin;
    glm::vec3 direction;
    float tmin = 0.001f;
    float tmax = 1e16f;
    MediumInstance current_medium;
};
}    // namespace atcg