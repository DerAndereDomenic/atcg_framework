#pragma once

#include <Core/glm.h>
#include <nanort.h>

namespace atcg
{
struct AccelerationStructureComponent
{
    AccelerationStructureComponent() = default;

    // Don't retrieve this from opengl each time used
    atcg::MemoryBuffer<glm::vec3> vertices;
    atcg::MemoryBuffer<glm::u32vec3> faces;
    nanort::BVHAccel<float> accel;
};

}    // namespace atcg