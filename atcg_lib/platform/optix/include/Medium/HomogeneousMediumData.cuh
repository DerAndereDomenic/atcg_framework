#pragma once

#include <Core/glm.h>

namespace atcg
{
struct HomogeneousMediumData
{
    glm::vec3* albedo;    // sigma_s / sigma_t
    float* density;       // sigma_t
    glm::vec3 Le;

    float* albedo_grad;
    float* density_grad;

    bool optimize_albedo  = false;
    bool optimize_density = false;
};
}    // namespace atcg