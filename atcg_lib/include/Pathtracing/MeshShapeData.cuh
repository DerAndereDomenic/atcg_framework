#pragma once

#include <Pathtracing/BSDFModels.cuh>
#include <Pathtracing/EmitterModels.cuh>

namespace atcg
{
struct MeshShapeData
{
    glm::vec3* positions;
    glm::vec3* normals;
    glm::vec3* uvs;
    glm::u32vec3* faces;

    uint32_t num_vertices;
    uint32_t num_faces;

    const BSDFVPtrTable* bsdf;
    const EmitterVPtrTable* emitter;
};
}    // namespace atcg