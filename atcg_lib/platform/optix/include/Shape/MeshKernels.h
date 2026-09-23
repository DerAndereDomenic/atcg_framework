#pragma once

#include <Core/glm.h>
#include <DataStructure/TorchUtils.h>

namespace atcg
{
void computeMeshTrianglePDFKernel(const torch::Tensor& positions,
                                  const torch::Tensor& indices,
                                  const glm::mat4& transform,
                                  torch::Tensor& pdf);

void computeMeshTriangleCDFKernel(const torch::Tensor& cdf);

void normalizeMeshTriangleCDFKernel(const torch::Tensor& cdf, float total_area);
}    // namespace atcg