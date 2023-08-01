#pragma once

#include <Core/Memory.h>

#include <Eigen/Dense>
#include <vector>

namespace atcg
{
namespace Tracing
{
/**
 * @brief Compute ray mesh intersections
 *
 * @param mesh The mesh
 * @param origin Vector with starting positions
 * @param direction Vector with starting directions
 * @param tmin The starting offset
 * @param tmax Maximum ray length
 *
 * @return vector with depth to the first intersection point
 */
// Eigen::VectorXf rayMeshIntersection(const atcg::ref_ptr<Mesh>& mesh,
//                                     const Eigen::MatrixX3f& origin,
//                                     const Eigen::MatrixX3f& direction,
//                                     const float tmin,
//                                     const float tmax);
}    // namespace Tracing
}    // namespace atcg