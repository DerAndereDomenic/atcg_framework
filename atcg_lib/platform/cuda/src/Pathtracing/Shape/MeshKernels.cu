#pragma cuda_source_property_format = PTX

#include <Pathtracing/SurfaceInteraction.h>
#include <Pathtracing/Shape/MeshShape.cuh>
#include <Pathtracing/Payload.h>

#include <optix.h>

extern "C" __global__ void __closesthit__mesh()
{
    atcg::SurfaceInteraction* si = getPayloadDataPointer<atcg::SurfaceInteraction>();
    const MeshShapeData sbt_data = *reinterpret_cast<const MeshShapeData*>(optixGetSbtDataPointer());

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

    si->bsdf    = sbt_data.bsdf;
    si->emitter = sbt_data.emitter;
}