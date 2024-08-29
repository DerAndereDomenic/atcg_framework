#pragma cuda_source_property_format = PTX

#include <iostream>
#include <Pathtracing/PathtracingPlatform.h>
#include <Pathtracing/MeshShapeData.cuh>

#include <Pathtracing/Stubs.h>

extern "C" ATCG_RT_EXPORT void __closesthit__mesh()
{
    atcg::SurfaceInteraction* si = getPayloadDataPointer<atcg::SurfaceInteraction>();

    // TODO
    atcg::MeshShapeData sbt_data = *reinterpret_cast<atcg::MeshShapeData*>(atcg::g_function_memory_map["__"
                                                                                                       "closesthit_"
                                                                                                       "_mesh"]);

    glm::vec3 ray_origin = getWorldRayOrigin();       // TODO
    glm::vec3 ray_dir    = getWorldRayDirection();    // TODO
    float tmax           = getRayTmax();              // TODO

    si->valid              = true;
    si->incoming_distance  = tmax;
    si->incoming_direction = ray_dir;
    si->primitive_idx      = getPrimitiveIndex();          // TODO
    si->barys              = getTriangleBarycentrics();    // TODO

    glm::u32vec3 triangle = sbt_data.faces[si->primitive_idx];

    const glm::vec3 P0 = sbt_data.positions[triangle.x];
    const glm::vec3 P1 = sbt_data.positions[triangle.y];
    const glm::vec3 P2 = sbt_data.positions[triangle.z];
    si->position       = (1.0f - si->barys.x - si->barys.y) * P0 + si->barys.x * P1 + si->barys.y * P2;
    // Transform local position to world position
    glm::vec3 optix_pos = transformPointFromObjectToWorldSpace(si->position);    // TODO
    si->position        = glm::make_vec3((float*)&optix_pos);

    const glm::vec3 N0 = sbt_data.normals[triangle.x];
    const glm::vec3 N1 = sbt_data.normals[triangle.y];
    const glm::vec3 N2 = sbt_data.normals[triangle.z];
    si->normal         = (1.0f - si->barys.x - si->barys.y) * N0 + si->barys.x * N1 + si->barys.y * N2;
    // Transform local position to world position
    si->normal = transformNormalFromObjectToWorldSpace(si->normal);    // TODO

    const glm::vec3 UV0 = sbt_data.uvs[triangle.x];
    const glm::vec3 UV1 = sbt_data.uvs[triangle.y];
    const glm::vec3 UV2 = sbt_data.uvs[triangle.z];
    si->uv              = (1.0f - si->barys.x - si->barys.y) * UV0 + si->barys.x * UV1 + si->barys.y * UV2;

    si->bsdf    = sbt_data.bsdf;
    si->emitter = sbt_data.emitter;
}