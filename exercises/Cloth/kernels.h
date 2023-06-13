#pragma once

#include <Core/Memory.h>
#include <Core/glm.h>

ATCG_HOST_DEVICE inline void updatePoint(glm::vec3* points, size_t tid, uint32_t j, uint32_t i, float time)
{
    points[tid].z = glm::sin(2.0f * glm::pi<float>() * (time) + j / 3.0f + i);
}

void simulate(glm::vec3* points, uint32_t size, float time);