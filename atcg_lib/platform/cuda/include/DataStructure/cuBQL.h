#pragma once

#include <cuBQL/bvh.h>
#include <Core/API.h>

namespace atcg
{
ATCG_API void build3fBVH(cuBQL::BinaryBVH<float, 3>& bvh,
                         /*! array of bounding boxes to build BVH over, must
                           be in device memory */
                         const cuBQL::box_t<float, 3>* boxes,
                         uint32_t numBoxes);

ATCG_API void free3fBVH(cuBQL::BinaryBVH<float, 3>& bvh);
}    // namespace atcg