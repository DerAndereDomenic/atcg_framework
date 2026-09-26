#pragma once

namespace atcg
{
struct MeshShapeData
{
    glm::vec3* positions;
    glm::vec3* normals;
    glm::vec3* colors;
    glm::vec3* uvs;
    glm::u32vec3* faces_3d;
    glm::u32vec3* faces_uv;
    glm::u32vec3* faces_normals;
    glm::u32vec3* faces_color;
};
}    // namespace atcg