#pragma once

#include <Scene/Entity.h>

#include <nanort.h>

namespace atcg
{
struct SurfaceInteraction
{
    bool valid = false;
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 barys;
    glm::vec2 uv;
    glm::vec3 incoming_direction;
    float incoming_distance;
    uint32_t primitive_idx;
};

namespace Tracing
{

struct HitInfo
{
    bool hit = false;
    glm::vec3 p;
    uint32_t primitive_idx;
};

/**
 * @brief Prepare the acceleration structure.
 * If it does not already have a AccelerationStructureComponent, a new one is created.
 * The entity needs to have a GeometryComponent
 *
 * @param entity The entity to prepare the BVH structure for
 *
 */
void prepareAccelerationStructure(Entity entity);

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
HitInfo traceRay(Entity entity, const glm::vec3& ray_origin, const glm::vec3& ray_dir, float t_min, float t_max);

/**
 * @brief Trace a ray against a mesh
 *
 * @param accel The bounding volume hierarchy
 * @param positions The positions that were used to build the BVH
 * @param normals The vertex normals
 * @param uvs The vertex uvs
 * @param faces The face index list that was used to build the BVH
 * @param origin The ray origin
 * @param dir The ray direction
 * @param tmin The minimal ray length
 * @param tmax The maximal ray length
 *
 * @return The intersection information
 */
SurfaceInteraction traceRay(const nanort::BVHAccel<float>& accel,
                            const std::vector<glm::vec3>& positions,
                            const std::vector<glm::vec3>& normals,
                            const std::vector<glm::vec3>& uvs,
                            const std::vector<glm::u32vec3>& faces,
                            const glm::vec3& origin,
                            const glm::vec3& dir,
                            float tmin,
                            float tmax);

}    // namespace Tracing
}    // namespace atcg