#pragma once

#include <Core/API.h>
#include <Scene/Entity.h>

namespace atcg
{
namespace Tracing
{

struct HitInfo
{
    glm::vec3 position;
    glm::vec3 incoming_direction;
    float incoming_distance;
    float pdf;

    ATCG_HOST_DEVICE HitInfo()
        : position(glm::vec3(std::numeric_limits<float>::signaling_NaN())),
          incoming_direction(glm::vec3(std::numeric_limits<float>::signaling_NaN())),
          incoming_distance(std::numeric_limits<float>::signaling_NaN()),
          pdf(0.0f)
    {
    }

    glm::vec3 normal;
    glm::vec2 barys;
    glm::vec2 uv;
    uint32_t primitive_idx;
    uint32_t entity_id;

    glm::vec3 dx_du, dx_dv;

    ATCG_INLINE ATCG_HOST_DEVICE bool isValid() const { return !glm::isnan(incoming_distance); }

    ATCG_INLINE ATCG_HOST_DEVICE void setInvalid() { incoming_distance = std::numeric_limits<float>::signaling_NaN(); }

    ATCG_INLINE ATCG_HOST_DEVICE bool isFinite() const { return glm::isfinite(incoming_distance); }

    ATCG_INLINE ATCG_HOST_DEVICE void setInfinite() { incoming_distance = std::numeric_limits<float>::infinity(); }
};

/**
 * @brief Prepare the acceleration structure.
 * If it does not already have a AccelerationStructureComponent, a new one is created.
 * The entity needs to have a GeometryComponent
 *
 * @param entity The entity to prepare the BVH structure for
 *
 */
ATCG_API void prepareAccelerationStructure(Entity entity);

/**
 * @brief Trace a ray agains the geometry
 *
 * @param entity The entity. Needs to have a AccelerationStructureComponent that was created using
 * prepareAccelerationStructure.
 * @param ray_origin The ray origin
 * @param ray_dir The normalized ray direction
 * @param t_min The start of the ray
 * @param t_max Th end of the ray
 *
 * @return HitInfo of the intersection
 */
ATCG_API HitInfo
traceRay(Entity entity, const glm::vec3& ray_origin, const glm::vec3& ray_dir, float t_min, float t_max);


}    // namespace Tracing
}    // namespace atcg