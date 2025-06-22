#pragma once

#include <Shape/ShapeData.cuh>
#include <BSDF/BSDFVPtrTable.cuh>
#include <Emitter/EmitterVPtrTable.cuh>

namespace atcg
{
struct ShapeInstanceData
{
    ShapeData* shape;
    const BSDFVPtrTable* bsdf;
    const EmitterVPtrTable* emitter;
    uint32_t entity_id;
    glm::vec3 color;
};
}    // namespace atcg