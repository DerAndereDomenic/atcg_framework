#pragma once

#include <Pathtracing/BSDFModels.cuh>
#include <Pathtracing/EmitterModels.cuh>

struct HitGroupData
{
    glm::vec3* positions;
    glm::vec3* normals;
    glm::vec3* uvs;
    glm::u32vec3* faces;

    const atcg::BSDFVPtrTable* bsdf;
    const atcg::EmitterVPtrTable* emitter;
};