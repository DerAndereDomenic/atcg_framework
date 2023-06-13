#pragma once

#include <Core/Memory.h>
#include <Core/glm.h>

#ifdef ATCG_CUDA_BACKEND
void simulate(glm::vec3* points, uint32_t size, float time);
#else
// Just some hackery so that the github actions run through if CUDA is not used
void simulate(glm::vec3* points, uint32_t size, float time) {}
#endif