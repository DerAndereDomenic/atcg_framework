#pragma once

#include <Core/Memory.h>
#include <Core/glm.h>
#include <DataStructure/Graph.h>
#include <DataStructure/BufferView.h>

inline ATCG_HOST_DEVICE void
updatePoint(atcg::BufferView<glm::vec3>& points, size_t tid, uint32_t j, uint32_t i, float time)
{
    points[tid].z = glm::sin(2.0f * glm::pi<float>() * (time) + j / 3.0f + i);
}

void simulate(atcg::BufferView<glm::vec3>& points, uint32_t size, float time);