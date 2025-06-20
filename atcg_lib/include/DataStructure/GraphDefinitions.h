#pragma once

#include <Core/glm.h>
#include <Core/Platform.h>
#include <torch/types.h>

namespace atcg
{

/**
 * @brief A struct that specifies how vertices are represented as a raw float buffer.
 * This is useful for filling and defining vertex data as tensors.
 *
 * {name}_BEGIN variables state the index of the first float where the {name} attribute starts.
 * {name}_END denotes the first index that now longer is part of the {name} attribute.
 *
 * Example:
 * vertices.index_put_({Slice(), Slice(POSITION_BEGIN, POSITION_END)}, position_data);
 *
 * VERTEX_SIZE is the total number of floats to describe one vertex
 */
struct VertexSpecification
{
    static ATCG_CONST ATCG_CONSTEXPR int POSITION_BEGIN = 0;
    static ATCG_CONST ATCG_CONSTEXPR int POSITION_END   = 3;

    static ATCG_CONST ATCG_CONSTEXPR int COLOR_BEGIN = 3;
    static ATCG_CONST ATCG_CONSTEXPR int COLOR_END   = 6;

    static ATCG_CONST ATCG_CONSTEXPR int NORMAL_BEGIN = 6;
    static ATCG_CONST ATCG_CONSTEXPR int NORMAL_END   = 9;

    static ATCG_CONST ATCG_CONSTEXPR int TANGENT_BEGIN = 9;
    static ATCG_CONST ATCG_CONSTEXPR int TANGENT_END   = 12;

    static ATCG_CONST ATCG_CONSTEXPR int UV_BEGIN = 12;
    static ATCG_CONST ATCG_CONSTEXPR int UV_END   = 15;

    static ATCG_CONST ATCG_CONSTEXPR int VERTEX_SIZE = 15;
};

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
 * @brief A struct that specifies how edges are represented as a raw float buffer.
 * This is useful for filling and defining edge data as tensors.
 *
 * {name}_BEGIN variables state the index of the first float where the {name} attribute starts.
 * {name}_END denotes the first index that now longer is part of the {name} attribute.
 *
 * Example:
 * edges.index_put_({Slice(), Slice(INDICES_BEGIN, INDICES_END)}, connection_data);
 *
 * EDGE_SIZE is the total number of floats to describe one vertex
 */
struct EdgeSpecification
{
    static ATCG_CONST ATCG_CONSTEXPR int INDICES_BEGIN = 0;
    static ATCG_CONST ATCG_CONSTEXPR int INDICES_END   = 2;

    static ATCG_CONST ATCG_CONSTEXPR int COLOR_BEGIN = 2;
    static ATCG_CONST ATCG_CONSTEXPR int COLOR_END   = 5;

    static ATCG_CONST ATCG_CONSTEXPR int RADIUS_BEGIN = 5;
    static ATCG_CONST ATCG_CONSTEXPR int RADIUS_END   = 6;

    static ATCG_CONST ATCG_CONSTEXPR int EDGE_SIZE = 6;
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
}    // namespace atcg