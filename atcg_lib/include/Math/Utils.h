#pragma once

#include <Core/Memory.h>
#include <DataStructure/Graph.h>
#include <Scene/Components.h>

#include <numeric>

namespace atcg
{
/**
 * @brief Normalizes a graph to the unit cupe
 *
 * @param graph The graph to normalize
 */
void normalize(const atcg::ref_ptr<Graph>& graph);

/**
 * @brief Apply a transform to a given mesh.
 * After this, the mesh vertices will be in world space and the transform will be the identity.
 *
 * @param graph The graph
 * @param transform The transform
 */
void applyTransform(const atcg::ref_ptr<Graph>& graph, atcg::TransformComponent& transform);

}    // namespace atcg