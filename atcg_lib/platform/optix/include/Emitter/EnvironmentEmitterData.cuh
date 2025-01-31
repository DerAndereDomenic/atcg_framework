#pragma once

#include <Core/CUDA.h>

namespace atcg
{
struct EnvironmentEmitterData
{
    cudaTextureObject_t environment_texture;
};
}    // namespace atcg