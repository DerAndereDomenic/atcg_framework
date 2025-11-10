#pragma cuda_source_property_format = PTX

#include <Core/SurfaceInteraction.h>
#include <Shape/ShapeInstanceData.cuh>
#include <Shape/MeshShapeData.cuh>
#include <Core/Payload.h>

#include <optix.h>

#include <CuDiff/ext/glm/Traits.h>
#include <CuDiff/ext/glm/Function.h>

extern "C" __global__ void __closesthit__mesh()
{
    atcg::SurfaceInteraction* si = getPayloadDataPointer<atcg::SurfaceInteraction>();
    const atcg::ShapeInstanceData _sbt_data =
        *reinterpret_cast<const atcg::ShapeInstanceData*>(optixGetSbtDataPointer());
    const atcg::MeshShapeData sbt_data = *(atcg::MeshShapeData*)(_sbt_data.shape);

    float3 optix_world_origin = optixGetWorldRayOrigin();
    float3 optix_world_dir    = optixGetWorldRayDirection();
    glm::vec3 ray_origin      = glm::make_vec3((float*)&optix_world_origin);
    glm::vec3 ray_dir         = glm::make_vec3((float*)&optix_world_dir);
    float tmax                = optixGetRayTmax();

    si->valid              = true;
    si->incoming_distance  = tmax;
    si->incoming_direction = ray_dir;
    si->primitive_idx      = optixGetPrimitiveIndex();
    float2 optix_barys     = optixGetTriangleBarycentrics();
    si->barys              = glm::make_vec2((float*)&optix_barys);

    glm::u32vec3 triangle = sbt_data.faces[si->primitive_idx];

    const glm::vec3 P0 = sbt_data.positions[triangle.x];
    const glm::vec3 P1 = sbt_data.positions[triangle.y];
    const glm::vec3 P2 = sbt_data.positions[triangle.z];
    si->position       = (1.0f - si->barys.x - si->barys.y) * P0 + si->barys.x * P1 + si->barys.y * P2;
    // Transform local position to world position
    float3 optix_pos =
        optixTransformPointFromObjectToWorldSpace(make_float3(si->position.x, si->position.y, si->position.z));
    si->position = glm::make_vec3((float*)&optix_pos);

    const glm::vec3 N0 = sbt_data.normals[triangle.x];
    const glm::vec3 N1 = sbt_data.normals[triangle.y];
    const glm::vec3 N2 = sbt_data.normals[triangle.z];
    si->normal         = (1.0f - si->barys.x - si->barys.y) * N0 + si->barys.x * N1 + si->barys.y * N2;
    // Transform local position to world position
    float3 optix_normal =
        optixTransformNormalFromObjectToWorldSpace(make_float3(si->normal.x, si->normal.y, si->normal.z));
    si->normal = glm::normalize(glm::make_vec3((float*)&optix_normal));

    const glm::vec3 UV0 = sbt_data.uvs[triangle.x];
    const glm::vec3 UV1 = sbt_data.uvs[triangle.y];
    const glm::vec3 UV2 = sbt_data.uvs[triangle.z];
    si->uv              = (1.0f - si->barys.x - si->barys.y) * UV0 + si->barys.x * UV1 + si->barys.y * UV2;

    const glm::vec3 C0 = sbt_data.colors[triangle.x];
    const glm::vec3 C1 = sbt_data.colors[triangle.y];
    const glm::vec3 C2 = sbt_data.colors[triangle.z];
    si->color          = (1.0f - si->barys.x - si->barys.y) * C0 + si->barys.x * C1 + si->barys.y * C2;
    si->color *= _sbt_data.color;

    si->bsdf    = _sbt_data.bsdf;
    si->emitter = _sbt_data.emitter;

    si->entity_id      = _sbt_data.entity_id;
    si->inside_medium  = _sbt_data.inside_medium;
    si->outside_medium = _sbt_data.outside_medium;
}

extern "C" __global__ void __closesthit__dual_mesh()
{
    atcg::DualSurfaceInteraction* si = getPayloadDataPointer<atcg::DualSurfaceInteraction>();
    const atcg::ShapeInstanceData _sbt_data =
        *reinterpret_cast<const atcg::ShapeInstanceData*>(optixGetSbtDataPointer());
    const atcg::MeshShapeData sbt_data = *(atcg::MeshShapeData*)(_sbt_data.shape);

    float3 optix_world_origin = optixGetWorldRayOrigin();
    float3 optix_world_dir    = optixGetWorldRayDirection();
    glm::vec3 ray_origin_     = glm::make_vec3((float*)&optix_world_origin);
    glm::vec3 ray_dir_        = glm::make_vec3((float*)&optix_world_dir);
    si->primitive_idx         = optixGetPrimitiveIndex();

    auto [xi, wi] = CuDiff::make_variables<6>(ray_origin_, ray_dir_);

    glm::u32vec3 triangle = sbt_data.faces[si->primitive_idx];
    const float3 P0_      = glm2cuda(sbt_data.positions[triangle.x]);
    const float3 P1_      = glm2cuda(sbt_data.positions[triangle.y]);
    const float3 P2_      = glm2cuda(sbt_data.positions[triangle.z]);

    // TODO: Kind of inefficient
    glm::vec3 P0 = cuda2glm(optixTransformPointFromObjectToWorldSpace(P0_));
    glm::vec3 P1 = cuda2glm(optixTransformPointFromObjectToWorldSpace(P1_));
    glm::vec3 P2 = cuda2glm(optixTransformPointFromObjectToWorldSpace(P2_));

    glm::vec3 v0              = P1 - P0;
    glm::vec3 v1              = P2 - P0;
    glm::vec3 geometry_normal = glm::normalize(glm::cross(v0, v1));

    auto tmax = (CuDiff::dot((P0 - xi), geometry_normal)) / (CuDiff::dot(wi, geometry_normal));
    auto xo   = xi + tmax * wi;

    // Calculate barycentric coordinates
    auto v2  = xo - P0;
    auto d00 = glm::dot(v0, v0);
    auto d01 = glm::dot(v0, v1);
    auto d11 = glm::dot(v1, v1);
    auto d20 = CuDiff::dot(v2, v0);
    auto d21 = CuDiff::dot(v2, v1);

    auto den = d00 * d11 - d01 * d01;

    auto beta  = (d11 * d20 - d01 * d21) / den;
    auto gamma = (d00 * d21 - d01 * d20) / den;
    auto alpha = 1.0f - beta - gamma;

    si->valid              = true;
    si->position           = xo;
    si->incoming_distance  = tmax;
    si->incoming_direction = wi;
    // float2 optix_barys     = optixGetTriangleBarycentrics();

    // Already global
    si->position = alpha * P0 + beta * P1 + gamma * P2;

    const float3 N0_ = glm2cuda(sbt_data.normals[triangle.x]);
    const float3 N1_ = glm2cuda(sbt_data.normals[triangle.y]);
    const float3 N2_ = glm2cuda(sbt_data.normals[triangle.z]);

    // TODO: Kind of inefficient
    const glm::vec3 N0 = cuda2glm(optixTransformNormalFromObjectToWorldSpace(N0_));
    const glm::vec3 N1 = cuda2glm(optixTransformNormalFromObjectToWorldSpace(N1_));
    const glm::vec3 N2 = cuda2glm(optixTransformNormalFromObjectToWorldSpace(N2_));
    si->normal         = CuDiff::normalize(alpha * N0 + beta * N1 + gamma * N2);

    const glm::vec3 UV0 = sbt_data.uvs[triangle.x];
    const glm::vec3 UV1 = sbt_data.uvs[triangle.y];
    const glm::vec3 UV2 = sbt_data.uvs[triangle.z];
    si->uv              = alpha * UV0 + beta * UV1 + gamma * UV2;

    // TODO
    // const glm::vec3 C0 = sbt_data.colors[triangle.x];
    // const glm::vec3 C1 = sbt_data.colors[triangle.y];
    // const glm::vec3 C2 = sbt_data.colors[triangle.z];
    // si->color          = (1.0f - si->barys.x - si->barys.y) * C0 + si->barys.x * C1 + si->barys.y * C2;
    // si->color *= _sbt_data.color;

    si->bsdf    = _sbt_data.bsdf;
    si->emitter = _sbt_data.emitter;

    si->entity_id      = _sbt_data.entity_id;
    si->inside_medium  = _sbt_data.inside_medium;
    si->outside_medium = _sbt_data.outside_medium;
}