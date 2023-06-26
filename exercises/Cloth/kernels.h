#pragma once

#include <Core/Memory.h>
#include <Core/glm.h>
#include <DataStructure/Graph.h>

inline ATCG_HOST_DEVICE void updatePoint(atcg::Vertex* points, size_t tid, uint32_t j, uint32_t i, float time)
{
    points[tid].position.z = glm::sin(2.0f * glm::pi<float>() * (time) + j / 3.0f + i);
}

void simulate(atcg::Vertex* points, uint32_t size, float time);