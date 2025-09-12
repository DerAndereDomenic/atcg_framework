#pragma once

#include <Core/glm.h>

namespace atcg
{
struct HomogeneousMediumData
{
    glm::vec3 sigma_a;
    glm::vec3 sigma_s;
    glm::vec3 Le;
};
}    // namespace atcg