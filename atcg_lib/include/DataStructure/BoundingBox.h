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

ATCG_INLINE BoundingBox operator+(const BoundingBox& a, const BoundingBox& b)
{
    BoundingBox result;
    result.min = glm::min(a.min, b.min);
    result.max = glm::max(a.max, b.max);
    return result;
}

ATCG_INLINE BoundingBox transformBoundingBox(const BoundingBox& bbox, const glm::mat4& transform)
{
    glm::vec3 min = transform * glm::vec4(bbox.min, 1.0f);
    glm::vec3 max = transform * glm::vec4(bbox.max, 1.0f);

    BoundingBox result;
    result.min = glm::min(min, max);
    result.max = glm::max(min, max);
    return result;
}

ATCG_INLINE glm::mat4 boundingBoxToModelMatrix(const BoundingBox& bbox)
{
    glm::vec3 center = (bbox.min + bbox.max) * 0.5f;
    glm::vec3 scale  = bbox.max - bbox.min;

    glm::mat4 model = glm::translate(center) * glm::scale(scale);
    return model;
}
}    // namespace atcg