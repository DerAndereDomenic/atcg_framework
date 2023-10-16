#pragma once

#include <Core/Memory.h>
#include <DataStructure/Graph.h>

#include <numeric>

namespace atcg
{
/**
 * @brief Normalizes a graph to the unit cupe
 *
 * @param graph The graph to normalize
 */
void normalize(const atcg::ref_ptr<Graph>& graph);

inline glm::mat3 compute_local_frame(const glm::vec3& localZ)
{
    float x  = localZ.x;
    float y  = localZ.y;
    float z  = localZ.z;
    float sz = (z >= 0) ? 1 : -1;
    float a  = 1 / (sz + z);
    float ya = y * a;
    float b  = x * ya;
    float c  = x * sz;

    glm::vec3 localX = glm::vec3(c * x * a - 1, sz * b, c);
    glm::vec3 localY = glm::vec3(b, y * ya - sz, y);

    glm::mat3 frame;
    // Set columns of matrix
    frame[0] = localX;
    frame[1] = localY;
    frame[2] = localZ;
    return frame;
}


}    // namespace atcg