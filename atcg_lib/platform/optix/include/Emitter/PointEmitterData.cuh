#pragma once

#include <Core/glm.h>

namespace atcg
{
struct PointEmitterData
{
    glm::vec3 position;
    float intensity;
    glm::vec3 color;
};
}    // namespace atcg