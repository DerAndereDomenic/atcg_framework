#pragma once

#include <Core/glm.h>

namespace atcg
{
struct MeshSamplerData
{
    glm::vec3* positions;
    glm::vec3* uvs;
    glm::u32vec3* faces_3d;
    glm::u32vec3* faces_uv;
    uint32_t num_faces;

    glm::mat4 world_to_local;
    glm::mat4 local_to_world;
    float* mesh_cdf;
    float total_area;
};
}    // namespace atcg