#pragma once

#include <Shape/ShapeData.cuh>
#include <BSDF/BSDFVPtrTable.cuh>

namespace atcg
{
struct ShapeInstanceData
{
    ShapeData* shape;
    const BSDFVPtrTable* bsdf;
};
}    // namespace atcg