#pragma once

#include <Core/glm.h>
#include <DataStructure/TorchUtils.h>

namespace atcg
{
torch::Tensor
computeMeshTriangleAreas(const torch::Tensor& positions, const torch::Tensor& indices, const glm::mat4& transform);

torch::Tensor
computeMeshEdgeLengths(const torch::Tensor& positions, const torch::Tensor& edges, const glm::mat4& transform);
}    // namespace atcg