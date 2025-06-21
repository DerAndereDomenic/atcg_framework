#pragma once

#include <Shape/ShapeData.cuh>
#include <BSDF/BSDFVPtrTable.cuh>

namespace atcg
{
struct ShapeInstanceData
{
    ShapeData* shape;
    const BSDFVPtrTable* bsdf;
    uint32_t entity_id;
    glm::vec3 color;
};
}    // namespace atcg