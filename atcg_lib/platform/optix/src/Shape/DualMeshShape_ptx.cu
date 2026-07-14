#pragma cuda_source_property_format = PTX

#include <Core/SurfaceInteraction.h>
#include <Shape/ShapeInstanceData.cuh>
#include <Shape/MeshShapeData.cuh>
#include <Core/Payload.h>

#include <optix.h>

#include <CuDiff/ext/glm/Traits.h>
#include <CuDiff/ext/glm/Function.h>

extern "C" __global__ void __closesthit__dual_mesh()
{
    atcg::DualSurfaceInteraction* si = getPayloadDataPointer<atcg::DualSurfaceInteraction>();
    const atcg::ShapeInstanceData _sbt_data =
        *reinterpret_cast<const atcg::ShapeInstanceData*>(optixGetSbtDataPointer());
    const atcg::MeshShapeData sbt_data = *(atcg::MeshShapeData*)(_sbt_data.shape);

    si->primitive_idx = optixGetPrimitiveIndex();

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

    auto xi = si->incoming_position;
    auto wi = si->incoming_direction;

    auto tmax = (CuDiff::dot((P0 - xi), geometry_normal)) / (CuDiff::dot(wi, geometry_normal));
    auto xo   = xi + tmax * wi;

    si->position          = xo;
    si->incoming_distance = tmax;
    // float2 optix_barys     = optixGetTriangleBarycentrics();

    // Calculate derivatives wrt to intersection point
    xo.setDerivative(0, glm::vec3(0.0f, 0.0f, 0.0f));
    xo.setDerivative(1, glm::vec3(0.0f, 0.0f, 0.0f));
    xo.setDerivative(2, glm::vec3(0.0f, 0.0f, 0.0f));
    xo.setDerivative(3, glm::vec3(1.0f, 0.0f, 0.0f));
    xo.setDerivative(4, glm::vec3(0.0f, 1.0f, 0.0f));
    xo.setDerivative(5, glm::vec3(0.0f, 0.0f, 1.0f));
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

    const float3 N0_ = glm2cuda(sbt_data.normals[triangle.x]);
    const float3 N1_ = glm2cuda(sbt_data.normals[triangle.y]);
    const float3 N2_ = glm2cuda(sbt_data.normals[triangle.z]);

    // TODO: Kind of inefficient
    const glm::vec3 N0  = cuda2glm(optixTransformNormalFromObjectToWorldSpace(N0_));
    const glm::vec3 N1  = cuda2glm(optixTransformNormalFromObjectToWorldSpace(N1_));
    const glm::vec3 N2  = cuda2glm(optixTransformNormalFromObjectToWorldSpace(N2_));
    si->normal          = CuDiff::normalize(alpha * N0 + beta * N1 + gamma * N2);
    si->reference_frame = atcg::Frame<glm::vec3>(si->normal.val());

    const glm::vec2 UV0 = sbt_data.uvs[triangle.x];
    const glm::vec2 UV1 = sbt_data.uvs[triangle.y];
    const glm::vec2 UV2 = sbt_data.uvs[triangle.z];
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

    si->dxdw = tmax.val() * glm::mat3(1.0f) -
               tmax.val() / glm::dot(geometry_normal, wi.val()) * (glm::outerProduct(wi.val(), geometry_normal));

    glm::mat3 dx1_dx0 = glm::mat3(0);
    glm::mat3 dx2_dx0 = glm::mat3(si->position.derivative(0), si->position.derivative(1), si->position.derivative(2));
    glm::mat3 dx1_dx1 = glm::mat3(1);
    glm::mat3 dx2_dx1 = glm::mat3(si->position.derivative(3), si->position.derivative(4), si->position.derivative(5));

    si->dx1x2_dx0x1 = atcg::mat6(dx1_dx0, dx1_dx1, dx2_dx0, dx2_dx1);
}