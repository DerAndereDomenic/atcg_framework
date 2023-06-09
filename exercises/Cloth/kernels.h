#pragma once

#include <Core/Memory.h>
#include <glm/glm.hpp>

void simulate(const atcg::ref_ptr<glm::vec3, atcg::device_allocator>& points, float time);