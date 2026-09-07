#pragma once

#include <Core/glm.h>
#include <nanort.h>

namespace atcg
{
struct ATCG_API AccelerationStructureComponent
{
    AccelerationStructureComponent() = default;

    // Don't retrieve this from opengl each time used
    torch::Tensor vertices;
    torch::Tensor faces;
    torch::Tensor normals;
    torch::Tensor uvs;

    nanort::BVHAccel<float> accel;
};

}    // namespace atcg