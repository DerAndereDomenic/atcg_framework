#pragma once

#include <Core/CUDA.h>
#include <DataStructure/TextureSampler.h>

namespace atcg
{
struct EnvironmentEmitterData
{
    TextureSampler<glm::vec3> environment_texture;
};
}    // namespace atcg