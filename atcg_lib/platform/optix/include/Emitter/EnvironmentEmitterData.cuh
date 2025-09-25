#pragma once

#include <Core/CUDA.h>
#include <DataStructure/CUDATexture.h>

namespace atcg
{
struct EnvironmentEmitterData
{
    CUDATexture<glm::vec3> environment_texture;
};
}    // namespace atcg