#pragma once

#include <Core/glm.h>
#include <Core/Platform.h>
#include <Pathtracing/PathtracingPlatform.h>
#include <Pathtracing/ShaderBindingTableDefinition.h>

#include <unordered_map>

namespace atcg
{

ATCGShaderBindingTable g_sbt = nullptr;

static thread_local std::unordered_map<std::string, uint8_t*> g_function_memory_map;
static thread_local void* payload_ptr;
static thread_local glm::vec3 ray_origin;
static thread_local glm::vec3 ray_direction;
static thread_local float ray_tmin;
static thread_local float ray_tmax;
static thread_local int primitive_idx;
static thread_local glm::mat4 local_to_world;
static thread_local glm::vec2 barys;
static thread_local uint32_t x;
static thread_local uint32_t y;

}    // namespace atcg

extern "C" ATCG_RT_EXPORT void set_sbt(atcg::ATCGShaderBindingTable sbt)
{
    atcg::g_sbt = sbt;
}

extern "C" ATCG_RT_EXPORT void set_memory_for_function(const char* func_name, uint8_t* memory)
{
    atcg::g_function_memory_map[func_name] = memory;
}

extern "C" ATCG_RT_EXPORT void set_payload_pointer(void* ptr)
{
    atcg::payload_ptr = ptr;
}

extern "C" ATCG_RT_EXPORT void set_ray_origin(const glm::vec3& origin)
{
    atcg::ray_origin = origin;
}

extern "C" ATCG_RT_EXPORT void set_ray_direction(const glm::vec3& direction)
{
    atcg::ray_direction = direction;
}

extern "C" ATCG_RT_EXPORT void set_ray_tmin(const float tmin)
{
    atcg::ray_tmin = tmin;
}

extern "C" ATCG_RT_EXPORT void set_ray_tmax(const float tmax)
{
    atcg::ray_tmax = tmax;
}

extern "C" ATCG_RT_EXPORT void set_primitive_idx(const int idx)
{
    atcg::primitive_idx = idx;
}

extern "C" ATCG_RT_EXPORT void set_local_to_world(const glm::mat4& transform)
{
    atcg::local_to_world = transform;
}

extern "C" ATCG_RT_EXPORT void set_barys(const glm::vec2& barys)
{
    atcg::barys = barys;
}

extern "C" ATCG_RT_EXPORT void set_pixel_index(const uint32_t px, const uint32_t py)
{
    atcg::x = px;
    atcg::y = py;
}

template<typename T>
ATCG_HOST ATCG_HOST_DEVICE T* getPayloadDataPointer()
{
    return static_cast<T*>(atcg::payload_ptr);
}

ATCG_HOST ATCG_HOST_DEVICE glm::vec3 getWorldRayOrigin()
{
    return atcg::ray_origin;
}

ATCG_HOST ATCG_HOST_DEVICE glm::vec3 getWorldRayDirection()
{
    return atcg::ray_direction;
}

ATCG_HOST ATCG_HOST_DEVICE float getRayTmax()
{
    return atcg::ray_tmax;
}

ATCG_HOST ATCG_HOST_DEVICE int getPrimitiveIndex()
{
    return atcg::primitive_idx;
}

ATCG_HOST ATCG_HOST_DEVICE glm::vec2 getTriangleBarycentrics()
{
    return atcg::barys;
}

ATCG_HOST ATCG_HOST_DEVICE glm::vec3 transformPointFromObjectToWorldSpace(const glm::vec3& x)
{
    return atcg::local_to_world * glm::vec4(x, 1);
}

ATCG_HOST ATCG_HOST_DEVICE glm::vec3 transformNormalFromObjectToWorldSpace(const glm::vec3& n)
{
    return glm::normalize(glm::inverse(glm::transpose(atcg::local_to_world)) * glm::vec4(n, 0));
}