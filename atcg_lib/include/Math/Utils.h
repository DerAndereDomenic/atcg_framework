#pragma once

#include <Core/Memory.h>
#include <DataStructure/Graph.h>
#include <Scene/Components.h>

#include <numeric>

namespace atcg
{
/**
 * @brief Normalizes a graph to the unit cube
 *
 * @param graph The graph to normalize
 */
void normalize(const atcg::ref_ptr<Graph>& graph);

/**
 * @brief Normalizes a graph to the unit cube and writes the inverse transformation into a transform
 *
 * @param graph The graph to normalize
 * @param transform The transform component
 */
void normalize(const atcg::ref_ptr<Graph>& graph, atcg::TransformComponent& transform);

/**
 * @brief Apply a transform to a given mesh.
 * After this, the mesh vertices will be in world space and the transform will be the identity.
 *
 * @param graph The graph
 * @param transform The transform
 */
void applyTransform(const atcg::ref_ptr<Graph>& graph, atcg::TransformComponent& transform);

/**
 * @brief Apply a transform to a given mesh.
 * After this, the mesh vertices will be in world space and the transform will be the identity.
 *
 * @param positions The positions
 * @param normals The normals
 * @param tangents The tangents
 * @param transform The transform
 */
void applyTransform(torch::Tensor& positions,
                    torch::Tensor& normals,
                    torch::Tensor& tangents,
                    atcg::TransformComponent& transform);

namespace IO
{
/**
 * @brief Dump data as raw binary file to disk.
 *
 * @param path The path
 * @param data The data
 */
void dumpBinary(const std::string& path, const torch::Tensor& data);
}    // namespace IO

}    // namespace atcg