#pragma once

#include <Shape/ShapeData.cuh>
#include <Material/BSDFVPtrTable.h>
#include <Emitter/EmitterVPtrTable.cuh>
#include <Material/MediumVPtrTable.h>

namespace atcg
{
struct ShapeInstanceData
{
    ShapeData* shape;
    const BSDFVPtrTable* bsdf;
    const EmitterVPtrTable* emitter;
    const MediumVPtrTable* inside_medium;
    const MediumVPtrTable* outside_medium;
    uint32_t entity_id;
    glm::vec3 color;
    glm::mat4 object_to_world;
    glm::mat4 world_to_object;
};
}    // namespace atcg