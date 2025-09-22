#pragma once

#include <Core/glm.h>

namespace atcg
{
/**
 * @brief Class to model an axis-aligned Bounding Box
 */
struct BoundingBox
{
    glm::vec3 min = glm::vec3(-1);
    glm::vec3 max = glm::vec3(1);
};
}    // namespace atcg