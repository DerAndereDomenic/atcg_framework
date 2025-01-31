#pragma once

namespace atcg
{
struct MeshShapeData
{
    glm::vec3* positions;
    glm::vec3* normals;
    glm::vec3* uvs;
    glm::u32vec3* faces;
};
}