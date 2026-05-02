#pragma once

#include <Core/glm.h>
#include <array>
#include <Renderer/Camera.h>

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

namespace Utils
{
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

ATCG_INLINE bool insideBoundingBox(const glm::vec3& point, const BoundingBox& bbox)
{
    return (point.x >= bbox.min.x && point.x <= bbox.max.x) && (point.y >= bbox.min.y && point.y <= bbox.max.y) &&
           (point.z >= bbox.min.z && point.z <= bbox.max.z);
}

ATCG_INLINE std::array<glm::vec3, 8> getBoundingBoxCorners(const BoundingBox& bbox)
{
    std::array<glm::vec3, 8> corners;

    corners[0] = glm::vec3(bbox.min.x, bbox.min.y, bbox.min.z);
    corners[1] = glm::vec3(bbox.max.x, bbox.min.y, bbox.min.z);
    corners[2] = glm::vec3(bbox.max.x, bbox.max.y, bbox.min.z);
    corners[3] = glm::vec3(bbox.min.x, bbox.max.y, bbox.min.z);
    corners[4] = glm::vec3(bbox.min.x, bbox.min.y, bbox.max.z);
    corners[5] = glm::vec3(bbox.max.x, bbox.min.y, bbox.max.z);
    corners[6] = glm::vec3(bbox.max.x, bbox.max.y, bbox.max.z);
    corners[7] = glm::vec3(bbox.min.x, bbox.max.y, bbox.max.z);

    return corners;
}

ATCG_INLINE bool isVisible(const atcg::ref_ptr<Camera>& camera, const BoundingBox& bbox)
{
    glm::mat4 view_projection = glm::transpose(camera->getViewProjection());
    std::array<glm::vec4, 6> planes;
    planes[0] = view_projection[3] + view_projection[0];    // left
    planes[1] = view_projection[3] - view_projection[0];    // right
    planes[2] = view_projection[3] + view_projection[1];    // bottom
    planes[3] = view_projection[3] - view_projection[1];    // top
    planes[4] = view_projection[3] + view_projection[2];    // near
    planes[5] = view_projection[3] - view_projection[2];    // far

    for(const auto& plane: planes)
    {
        glm::vec3 normal = glm::vec3(plane);
        float length     = glm::length(normal);
        float distance   = plane.w / length;
        normal           = normal / length;

        glm::vec3 positive_vertex = bbox.min;
        if(normal.x >= 0) positive_vertex.x = bbox.max.x;
        if(normal.y >= 0) positive_vertex.y = bbox.max.y;
        if(normal.z >= 0) positive_vertex.z = bbox.max.z;

        if((glm::dot(normal, positive_vertex) + distance) < 0)
        {
            return false;
        }
    }
    return true;
}

}    // namespace Utils
}    // namespace atcg