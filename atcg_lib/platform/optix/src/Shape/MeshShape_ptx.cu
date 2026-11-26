#pragma cuda_source_property_format = PTX

#include <Core/SurfaceInteraction.h>
#include <Shape/ShapeInstanceData.cuh>
#include <Shape/MeshShapeData.cuh>
#include <Core/Payload.h>
#include <DataStructure/Frame.h>

#include <optix.h>

extern "C" __global__ void __closesthit__mesh()
{
    atcg::SurfaceInteraction* si = getPayloadDataPointer<atcg::SurfaceInteraction>();
    const atcg::ShapeInstanceData* _sbt_data =
        *reinterpret_cast<const atcg::ShapeInstanceData**>(optixGetSbtDataPointer());
    const atcg::MeshShapeData sbt_data = *(atcg::MeshShapeData*)(_sbt_data->shape);

    float3 optix_world_dir = optixGetWorldRayDirection();
    glm::vec3 ray_dir      = glm::make_vec3((float*)&optix_world_dir);
    float tmax             = optixGetRayTmax();

    si->incoming_distance  = tmax;
    si->incoming_direction = ray_dir;
    si->primitive_idx      = optixGetPrimitiveIndex();
    float2 optix_barys     = optixGetTriangleBarycentrics();
    si->barys              = glm::make_vec2((float*)&optix_barys);

    glm::u32vec3 triangle_position_indices = sbt_data.faces_3d[si->primitive_idx];
    glm::u32vec3 triangle_uv_indices       = sbt_data.faces_uv[si->primitive_idx];
    glm::u32vec3 triangle_normal_indices   = sbt_data.faces_normals[si->primitive_idx];
    glm::u32vec3 triangle_color_indices    = sbt_data.faces_color[si->primitive_idx];
    glm::u32vec3 triangle_tangent_indices  = sbt_data.faces_tangents[si->primitive_idx];

    const glm::vec3 P0 = sbt_data.positions[triangle_position_indices.x];
    const glm::vec3 P1 = sbt_data.positions[triangle_position_indices.y];
    const glm::vec3 P2 = sbt_data.positions[triangle_position_indices.z];
    si->position       = (1.0f - si->barys.x - si->barys.y) * P0 + si->barys.x * P1 + si->barys.y * P2;
    // Transform local position to world position
    float3 optix_pos =
        optixTransformPointFromObjectToWorldSpace(make_float3(si->position.x, si->position.y, si->position.z));
    si->position = glm::make_vec3((float*)&optix_pos);

    const glm::vec3 N0 = sbt_data.normals[triangle_normal_indices.x];
    const glm::vec3 N1 = sbt_data.normals[triangle_normal_indices.y];
    const glm::vec3 N2 = sbt_data.normals[triangle_normal_indices.z];
    si->normal         = (1.0f - si->barys.x - si->barys.y) * N0 + si->barys.x * N1 + si->barys.y * N2;
    // Transform local position to world position
    float3 optix_normal =
        optixTransformNormalFromObjectToWorldSpace(make_float3(si->normal.x, si->normal.y, si->normal.z));
    si->normal = glm::normalize(glm::make_vec3((float*)&optix_normal));

    const glm::vec3 UV0 = sbt_data.uvs[triangle_uv_indices.x];
    const glm::vec3 UV1 = sbt_data.uvs[triangle_uv_indices.y];
    const glm::vec3 UV2 = sbt_data.uvs[triangle_uv_indices.z];
    si->uv              = (1.0f - si->barys.x - si->barys.y) * UV0 + si->barys.x * UV1 + si->barys.y * UV2;

    const glm::vec3 C0 = sbt_data.colors[triangle_color_indices.x];
    const glm::vec3 C1 = sbt_data.colors[triangle_color_indices.y];
    const glm::vec3 C2 = sbt_data.colors[triangle_color_indices.z];
    si->color          = (1.0f - si->barys.x - si->barys.y) * C0 + si->barys.x * C1 + si->barys.y * C2;
    si->color *= _sbt_data->color;

    si->bsdf    = _sbt_data->bsdf;
    si->emitter = _sbt_data->emitter;

    si->entity_id      = _sbt_data->entity_id;
    si->inside_medium  = _sbt_data->inside_medium;
    si->outside_medium = _sbt_data->outside_medium;

    if(sbt_data.tangents)
    {
        // Tangents from uv parameterization
        const glm::vec3 T0 = sbt_data.tangents[triangle_tangent_indices.x];
        const glm::vec3 T1 = sbt_data.tangents[triangle_tangent_indices.y];
        const glm::vec3 T2 = sbt_data.tangents[triangle_tangent_indices.z];

        glm::vec3 tangent = (1.0f - si->barys.x - si->barys.y) * T0 + si->barys.x * T1 + si->barys.y * T2;
        tangent           = glm::normalize(tangent);

        glm::vec3 bitangent = glm::normalize(glm::cross(si->normal, tangent));

        si->dx_du = tangent;
        si->dx_dv = bitangent;
    }
    else
    {
        // Fallback
        atcg::Frame frame = atcg::Frame(si->normal);    // Create a local frame based on the surface normal
        si->dx_du         = frame.localX();
        si->dx_dv         = frame.localY();
    }
}