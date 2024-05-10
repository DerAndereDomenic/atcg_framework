#pragma once

#include <Pathtracing/BSDF/BSDFModels.cuh>
#include <Pathtracing/Emitter/EmitterModels.cuh>

struct HitGroupData
{
    glm::vec3* positions;
    glm::vec3* normals;
    glm::vec3* uvs;
    glm::u32vec3* faces;

    const atcg::BSDFVPtrTable* bsdf;
    const atcg::EmitterVPtrTable* emitter;
};