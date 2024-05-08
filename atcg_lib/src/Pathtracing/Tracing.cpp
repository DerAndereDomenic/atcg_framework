#include <Pathtracing/Tracing.h>
#include <Scene/Components.h>
#include <Scene/Entity.h>

namespace atcg
{

void Tracing::prepareAccelerationStructure(Entity entity)
{
    if(!entity.hasComponent<GeometryComponent>())
    {
        ATCG_WARN("Entity does not have a geometry component. Cancel BVH build...");
        return;
    }

    if(!entity.hasComponent<AccelerationStructureComponent>())
    {
        entity.addComponent<AccelerationStructureComponent>();
    }

    auto& acc_component = entity.getComponent<AccelerationStructureComponent>();

    auto& geometry = entity.getComponent<GeometryComponent>();

    acc_component.accel = atcg::make_ref<BVHAccelerationStructure>(geometry.graph);
}

SurfaceInteraction
Tracing::traceRay(Entity entity, const glm::vec3& ray_origin, const glm::vec3& ray_dir, float t_min, float t_max)
{
    if(!entity.hasComponent<AccelerationStructureComponent>())
    {
        ATCG_WARN("Entity does not have an acceleration structure. Aborting...");
        return SurfaceInteraction();
    }

    auto& acc_component = entity.getComponent<AccelerationStructureComponent>();

    atcg::ref_ptr<BVHAccelerationStructure> bvh_accel = std::dynamic_pointer_cast<BVHAccelerationStructure>(
        entity.getComponent<AccelerationStructureComponent>().accel);

    if(!bvh_accel->getBVH().IsValid())
    {
        ATCG_WARN("Acceleration Structure not valid. Aborting...");
        return SurfaceInteraction();
    }

    return traceRay(bvh_accel,
                    bvh_accel->getPositions(),
                    bvh_accel->getNormals(),
                    bvh_accel->getUVs(),
                    bvh_accel->getFaces(),
                    ray_origin,
                    ray_dir,
                    t_min,
                    t_max);
}

SurfaceInteraction Tracing::traceRay(const atcg::ref_ptr<BVHAccelerationStructure>& accel,
                                     const torch::Tensor& positions,
                                     const torch::Tensor& normals,
                                     const torch::Tensor& uvs,
                                     const torch::Tensor& faces,
                                     const glm::vec3& origin,
                                     const glm::vec3& dir,
                                     float tmin,
                                     float tmax)
{
    if(!accel->getBVH().IsValid())
    {
        ATCG_WARN("Acceleration Structure not valid. Aborting...");
        return SurfaceInteraction();
    }

    SurfaceInteraction si;
    si.incoming_direction = dir;

    nanort::Ray<float> ray;
    memcpy(ray.org, glm::value_ptr(origin), sizeof(glm::vec3));
    memcpy(ray.dir, glm::value_ptr(dir), sizeof(glm::vec3));

    ray.min_t = tmin;
    ray.max_t = tmax;

    nanort::TriangleIntersector<> triangle_intersector(reinterpret_cast<const float*>(positions.data_ptr()),
                                                       reinterpret_cast<const uint32_t*>(faces.data_ptr()),
                                                       sizeof(float) * 3);
    nanort::TriangleIntersection<> isect;
    bool hit = accel->getBVH().Traverse(ray, triangle_intersector, &isect);

    if(!hit)
    {
        return si;
    }

    si.valid         = true;
    si.position      = origin + isect.t * dir;
    si.barys         = glm::vec2(isect.u, isect.v);
    si.primitive_idx = isect.prim_id;

    glm::u32vec3 face = glm::u32vec3(faces.index({(int)isect.prim_id, 0}).item<uint32_t>(),
                                     faces.index({(int)isect.prim_id, 1}).item<uint32_t>(),
                                     faces.index({(int)isect.prim_id, 2}).item<uint32_t>());

    glm::vec3 n1 = glm::vec3(normals.index({(int)face.x, 0}).item<float>(),
                             normals.index({(int)face.x, 1}).item<float>(),
                             normals.index({(int)face.x, 2}).item<float>());

    glm::vec3 n2 = glm::vec3(normals.index({(int)face.y, 0}).item<float>(),
                             normals.index({(int)face.y, 1}).item<float>(),
                             normals.index({(int)face.y, 2}).item<float>());

    glm::vec3 n3 = glm::vec3(normals.index({(int)face.z, 0}).item<float>(),
                             normals.index({(int)face.z, 1}).item<float>(),
                             normals.index({(int)face.z, 2}).item<float>());

    si.normal = (1.0f - isect.u - isect.v) * n1 + isect.u * n2 + isect.v * n3;

    glm::vec3 uv1 = glm::vec3(uvs.index({(int)face.x, 0}).item<float>(),
                              uvs.index({(int)face.x, 1}).item<float>(),
                              uvs.index({(int)face.x, 2}).item<float>());

    glm::vec3 uv2 = glm::vec3(uvs.index({(int)face.y, 0}).item<float>(),
                              uvs.index({(int)face.y, 1}).item<float>(),
                              uvs.index({(int)face.y, 2}).item<float>());

    glm::vec3 uv3 = glm::vec3(uvs.index({(int)face.z, 0}).item<float>(),
                              uvs.index({(int)face.z, 1}).item<float>(),
                              uvs.index({(int)face.z, 2}).item<float>());

    si.uv                = (1.0f - isect.u - isect.v) * uv1 + isect.u * uv2 + isect.v * uv3;
    si.incoming_distance = isect.t;

    return si;
}

}    // namespace atcg