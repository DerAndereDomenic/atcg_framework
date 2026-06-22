#pragma once

#include <Core/CUDA.h>
#include <DataStructure/CUDATexture.h>

namespace atcg
{
struct EnvironmentEmitterData
{
    CUDATexture<glm::vec3> environment_texture;

    float* col_pdfs;
    float* col_cdfs;
    float* row_pdf;
    float* row_cdf;

    int width;
    int height;
};
}    // namespace atcg