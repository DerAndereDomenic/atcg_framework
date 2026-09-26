#pragma once

#include <Core/CUDA.h>
#include <Core/SurfaceInteraction.h>

// Convenience functions to handle different integration domains
namespace atcg
{

ATCG_HOST_DEVICE ATCG_INLINE float G(const glm::vec3& x1, const glm::vec3& n1, const glm::vec3& x2, const glm::vec3& n2)
{
    float distance_squared = glm::length2(x2 - x1);
    float cos_theta_i      = glm::abs(glm::dot(n1, glm::normalize(x2 - x1)));
    float cos_theta_o      = glm::abs(glm::dot(n2, glm::normalize(x1 - x2)));

    return (cos_theta_i * cos_theta_o) / distance_squared;
}

ATCG_HOST_DEVICE ATCG_INLINE float G(const SurfaceInteraction& si, const SurfaceInteraction& si_next)
{
    return G(si.position, si.normal, si_next.position, si_next.normal);
}

ATCG_HOST_DEVICE ATCG_INLINE float dw_dA(const glm::vec3& x1, const glm::vec3& x2, const glm::vec3& n2)
{
    float distance_squared = glm::length2(x2 - x1);
    float cos_theta_o      = glm::abs(glm::dot(n2, glm::normalize(x1 - x2)));

    return cos_theta_o / distance_squared;
}

ATCG_HOST_DEVICE ATCG_INLINE float dw_dA(const SurfaceInteraction& si, const SurfaceInteraction& si_next)
{
    return dw_dA(si.position, si_next.position, si_next.normal);
}

ATCG_HOST_DEVICE ATCG_INLINE float dA_dw(const glm::vec3& x1, const glm::vec3& x2, const glm::vec3& n2)
{
    return 1.0f / dw_dA(x1, x2, n2);
}

ATCG_HOST_DEVICE ATCG_INLINE float dA_dw(const SurfaceInteraction& si, const SurfaceInteraction& si_next)
{
    return 1.0f / dw_dA(si, si_next);
}

template<typename T>
ATCG_HOST_DEVICE ATCG_INLINE T
solidAngleToArea(const T& value, const glm::vec3& x1, const glm::vec3& x2, const glm::vec3& n2)
{
    return value * dw_dA(x1, x2, n2);
}

template<typename T>
ATCG_HOST_DEVICE ATCG_INLINE T solidAngleToArea(const T& value,
                                                const SurfaceInteraction& si,
                                                const SurfaceInteraction& si_next)
{
    return solidAngleToArea(value, si.position, si_next.position, si_next.normal);
}

template<typename T>
ATCG_HOST_DEVICE ATCG_INLINE T
areaToSolidAngle(const T& value, const glm::vec3& x1, const glm::vec3& x2, const glm::vec3& n2)
{
    return value * dA_dw(x1, x2, n2);
}

template<typename T>
ATCG_HOST_DEVICE ATCG_INLINE T areaToSolidAngle(const T& value,
                                                const SurfaceInteraction& si,
                                                const SurfaceInteraction& si_next)
{
    return areaToSolidAngle(value, si.position, si_next.position, si_next.normal);
}
}    // namespace atcg