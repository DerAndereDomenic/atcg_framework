#pragma once

#include <Core/CUDA.h>
#include <DataStructure/CUDATexture.h>
#include <DataStructure/BoundingBox.h>

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

    atcg::BoundingBox bounding_box;    // This has to be set via emitter->setSceneAABB() manually
};
}    // namespace atcg