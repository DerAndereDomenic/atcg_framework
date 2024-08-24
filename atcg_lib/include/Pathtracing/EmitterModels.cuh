#pragma once

#include <Math/Random.h>
#include <Math/Functions.h>
#include <Pathtracing/SurfaceInteraction.h>
#include <Pathtracing/BSDFModels.cuh>
#include <Pathtracing/DirectCall.h>

// TODO: Has to be removed at some point
#include <torch/types.h>

namespace atcg
{
struct MeshEmitterData
{
    torch::Tensor positions;
    torch::Tensor uvs;
    torch::Tensor faces;
    uint32_t num_faces;

    float emitter_scaling;
    torch::Tensor emissive_texture;

    glm::mat4 world_to_local;
    glm::mat4 local_to_world;
    torch::Tensor mesh_cdf;
    float total_area;
};

struct EnvironmentEmitterData
{
    torch::Tensor environment_texture;
};

struct EmitterSamplingResult
{
    glm::vec3 direction_to_light;
    float distance_to_light;
    glm::vec3 normal_at_light;
    glm::vec3 radiance_weight_at_receiver;
    float sampling_pdf;
    glm::vec3 uvs;
};

struct EmitterVPtrTable
{
    uint32_t evalCallIndex;
    uint32_t sampleCallIndex;
    uint32_t evalPdfCallIndex;

#ifdef ATCG_RT_MODULE
    glm::vec3 evalLight(const SurfaceInteraction& si) const
    {
        return directCall<glm::vec3, const SurfaceInteraction&>(evalCallIndex, si);
    }

    EmitterSamplingResult sampleLight(const SurfaceInteraction& si, PCG32& rng) const
    {
        return directCall<EmitterSamplingResult, const SurfaceInteraction&, PCG32&>(sampleCallIndex, si, rng);
    }

    float evalLightSamplingPdf(const SurfaceInteraction& last_si, const SurfaceInteraction& si) const
    {
        return directCall<float, const SurfaceInteraction&, const SurfaceInteraction&>(evalPdfCallIndex, last_si, si);
    }
#endif
};

/**
 * @brief Eval a mesh emitter
 *
 * @param emissive_color The emissive color
 * @param emitter_scaling The scaling (intensity) of the emitter
 *
 * @return The radiance value
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 evalMeshEmitter(const glm::vec3& emissive_color,
                                                             const float emitter_scaling)
{
    return emissive_color * emitter_scaling;
}

/**
 * @brief Sample a mesh emitter
 *
 * @param si The surface interaction to sample from
 * @param mesh_cdf The cdf of the mesh triangles
 * @param positions Vertex positions
 * @param uvs Vertex uvs
 * @param faces Face data
 * @param num_faces The number of total faces
 * @param total_area The surface area of the mesh
 * @param local_to_world Local to world transform
 * @param world_to_local World to local transform
 * @param rng The rng
 *
 * @return The sampling result
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE atcg::EmitterSamplingResult sampleMeshEmitter(const atcg::SurfaceInteraction& si,
                                                                                 const float* mesh_cdf,
                                                                                 const glm::vec3* positions,
                                                                                 const glm::vec3* uvs,
                                                                                 const glm::u32vec3* faces,
                                                                                 const uint32_t num_faces,
                                                                                 const float total_area,
                                                                                 const glm::mat4& local_to_world,
                                                                                 const glm::mat4& world_to_local,
                                                                                 atcg::PCG32& rng)
{
    atcg::EmitterSamplingResult result;

    // Select the triangle to sample a direction from uniformly at random, proportional to its surface area
    uint32_t triangle_index = 0;
    // Sample the barycentric coordinates on the triangle uniformly.
    glm::vec2 triangle_barys = glm::vec2(0, 0);

    triangle_index = Math::binary_search(mesh_cdf, rng.next1d(), num_faces);

    triangle_barys = rng.next2d();
    // Mirror barys at diagonal line to cover a triangle instead of a square
    if(triangle_barys.x + triangle_barys.y > 1) triangle_barys = glm::vec2(1) - triangle_barys;


    // Compute the `light_position` using the triangle_index and the triangle_barys on the mesh:

    // Indices of triangle vertices in the mesh
    glm::u32vec3 vertex_indices = faces[triangle_index];

    // Vertex positions of selected triangle
    glm::vec3 P0 = positions[vertex_indices.x];
    glm::vec3 P1 = positions[vertex_indices.y];
    glm::vec3 P2 = positions[vertex_indices.z];

    glm::vec3 UV0 = uvs[vertex_indices.x];
    glm::vec3 UV1 = uvs[vertex_indices.y];
    glm::vec3 UV2 = uvs[vertex_indices.z];

    // Compute local position
    glm::vec3 local_light_position =
        (1.0f - triangle_barys.x - triangle_barys.y) * P0 + triangle_barys.x * P1 + triangle_barys.y * P2;
    // Transform local position to world position
    glm::vec3 light_position = glm::vec3(local_to_world * glm::vec4(local_light_position, 1));

    // Compute UVS
    glm::vec3 uv = (1.0f - triangle_barys.x - triangle_barys.y) * UV0 + triangle_barys.x * UV1 + triangle_barys.y * UV2;
    result.uvs   = uv;

    // Compute local normal
    glm::vec3 local_light_normal = glm::cross(P1 - P0, P2 - P0);
    // Normals are transformed by (A^-1)^T instead of A
    glm::vec3 light_normal = glm::normalize(glm::transpose(glm::mat3(world_to_local)) * local_light_normal);

    // Assemble sampling result
    result.sampling_pdf = 0;    // initialize with invalid sample

    // light source sampling
    result.direction_to_light       = glm::normalize(light_position - si.position);
    float distance_to_light_squared = glm::length2(light_position - si.position) + 1e-5f;
    result.distance_to_light        = glm::length(light_position - si.position) + 1e-5f;
    result.normal_at_light          = light_normal;

    float one_over_light_position_pdf  = total_area;
    float cos_theta_on_light           = glm::abs(glm::dot(result.direction_to_light, light_normal));
    float one_over_light_direction_pdf = one_over_light_position_pdf * cos_theta_on_light / distance_to_light_squared;


    // Probability of sampling this direction via light source sampling
    result.sampling_pdf = 1 / one_over_light_direction_pdf;

    return result;
}

/**
 * @brief Evaluate the pdf of a mesh emitter
 *
 * @param last_si The last surface interaction
 * @param total_area The mesh area
 * @param si The current surface interaction
 *
 * @return The pdf
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE float
evalMeshEmitterPDF(const SurfaceInteraction& last_si, const float total_area, const SurfaceInteraction& si)
{
    // We can assume that outgoing ray dir actually intersects the light source.

    // Some useful quantities
    glm::vec3 light_normal         = si.normal;
    glm::vec3 light_ray_dir        = glm::normalize(si.position - last_si.position);
    float light_ray_length_squared = glm::length2(si.position - last_si.position);

    // The probability of sampling any position on the surface of the mesh is the reciprocal of its surface area.
    float light_position_pdf = 1 / total_area;

    // Probability of sampling this direction via light source sampling
    float cos_theta_on_light  = glm::abs(glm::dot(light_ray_dir, light_normal));
    float light_direction_pdf = light_position_pdf * light_ray_length_squared / cos_theta_on_light;

    return light_direction_pdf;
}

/**
 * @brief Evaluate an environment emitter
 *
 * @param si The surface interaction
 *
 * @return The uv cooridnates to perform the texture lookup
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec2 evalEnvironmentEmitter(const SurfaceInteraction& si)
{
    glm::vec3 ray_dir = si.incoming_direction;

    float theta = std::acos(ray_dir.y) / glm::pi<float>();
    float phi   = (std::atan2(ray_dir.z, ray_dir.x) + glm::pi<float>()) / (2.0f * glm::pi<float>());

    glm::vec2 uv(phi, theta);

    return uv;
}

/**
 * @brief Sample an environment emitter
 *
 * @param si The surface interaction
 * @param rng The rng
 *
 * @return The sampling result
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE EmitterSamplingResult sampleEnvironmentEmitter(const SurfaceInteraction& si,
                                                                                  PCG32& rng)
{
    atcg::EmitterSamplingResult result;

    glm::vec3 random_dir = atcg::warp_square_to_hemisphere_cosine(rng.next2d());
    float pdf            = atcg::warp_square_to_hemisphere_cosine_pdf(random_dir);
    glm::mat3 frame      = atcg::Math::compute_local_frame(si.normal);

    random_dir = frame * random_dir;

    glm::vec3 ray_dir = si.incoming_direction;

    float theta = std::acos(ray_dir.y) / glm::pi<float>();
    float phi   = (std::atan2(ray_dir.z, ray_dir.x) + glm::pi<float>()) / (2.0f * glm::pi<float>());

    glm::vec3 uv(phi, theta, 0);

    result.distance_to_light = std::numeric_limits<float>::infinity();
    result.sampling_pdf      = pdf;
    result.uvs               = uv;

    return result;
}

/**
 * @brief Evaluate the pdf of an environment emitter
 *
 * @param last_si The last surface interaction
 * @param si The current surface interaction
 *
 * @return The pdf
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE float evalEnvironmentEmitterSamplingPdf(const SurfaceInteraction& last_si,
                                                                           const SurfaceInteraction& si)
{
    // We can assume that outgoing ray dir actually intersects the light source.

    // Probability of sampling this direction via light source sampling
    glm::mat3 local_frame     = atcg::Math::compute_local_frame(last_si.normal);
    glm::vec3 local_direction = glm::transpose(local_frame) * si.incoming_direction;

    return atcg::warp_square_to_hemisphere_cosine_pdf(local_direction);
}
}    // namespace atcg