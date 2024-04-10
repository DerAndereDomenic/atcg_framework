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


}    // namespace atcg