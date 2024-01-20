#pragma once

#include <Core/glm.h>

namespace atcg
{
/**
 * @brief A struct to model a vertex
 */
struct Vertex
{
    Vertex() = default;

    Vertex(const glm::vec3& position,
           const glm::vec3& color   = glm::vec3(1),
           const glm::vec3& normal  = glm::vec3(0),
           const glm::vec3& tangent = glm::vec3(0),
           const glm::vec3& uv      = glm::vec3(0))
        : position(position),
          color(color),
          normal(normal),
          tangent(tangent),
          uv(uv)
    {
    }

    glm::vec3 position = glm::vec3(0);
    glm::vec3 color    = glm::vec3(1);
    glm::vec3 normal   = glm::vec3(0);
    glm::vec3 tangent  = glm::vec3(0);
    glm::vec3 uv       = glm::vec3(0);
};

/**
 * @brief A struct to model an edge
 */
struct Edge
{
    glm::vec2 indices;
    glm::vec3 color;
    float radius;
};

/**
 * @brief A struct that holds instance information
 */
struct Instance
{
    glm::mat4 model = glm::mat4(1);
    glm::vec3 color = glm::vec3(1);
};
}    // namespace atcg