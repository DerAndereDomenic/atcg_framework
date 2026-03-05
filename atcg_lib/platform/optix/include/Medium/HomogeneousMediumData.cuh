#pragma once

#include <Core/glm.h>

namespace atcg
{
struct HomogeneousMediumData
{
    glm::vec3 albedo;    // sigma_s / sigma_t
    float density;       // sigma_t
    glm::vec3 Le;
};
}    // namespace atcg