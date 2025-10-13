#pragma once

#include <DataStructure/CUDATexture.h>

namespace atcg
{
struct MeshEmitterData
{
    glm::vec3* positions;
    glm::vec3* uvs;
    glm::u32vec3* faces;
    uint32_t num_faces;

    float emitter_scaling;
    CUDATexture<glm::vec3> emissive_texture;

    glm::mat4 world_to_local;
    glm::mat4 local_to_world;
    float* mesh_cdf;
    float total_area;
};
}    // namespace atcg