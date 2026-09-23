#pragma once

#include <DataStructure/CUDATexture.h>
#include <Shape/ShapeSamplerVPtrTable.cuh>

namespace atcg
{
struct MeshEmitterData
{
    float emitter_scaling;
    CUDATexture<glm::vec3> emissive_texture;

    const ShapeSamplerVPtrTable* sampler;
};
}    // namespace atcg