#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>
#include <Math/Functions.h>
#include <Core/SurfaceInteraction.h>

#include <Emitter/EmitterVPtrTable.cuh>
#include <Emitter/MeshEmitterData.cuh>

namespace detail
{
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
ATCG_HOST_DEVICE ATCG_FORCE_INLINE atcg::EmitterSamplingResult sampleMeshEmitter(const atcg::AnyInteraction& si,
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

    triangle_index = atcg::Math::binary_search(mesh_cdf, rng.next1d(), num_faces);

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
    result.direction_to_light       = glm::normalize(light_position - si->position);
    float distance_to_light_squared = glm::length2(light_position - si->position) + 1e-5f;
    result.distance_to_light        = glm::length(light_position - si->position) + 1e-5f;
    result.normal_at_light          = light_normal;

    float one_over_light_position_pdf  = total_area;
    float cos_theta_on_light           = glm::abs(glm::dot(result.direction_to_light, light_normal));
    float one_over_light_direction_pdf = one_over_light_position_pdf * cos_theta_on_light / distance_to_light_squared;


    // Probability of sampling this direction via light source sampling
    result.sampling_pdf = 1 / (one_over_light_direction_pdf + 1e-5f);

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
evalMeshEmitterPDF(const atcg::AnyInteraction& last_si, const float total_area, const atcg::SurfaceInteraction& si)
{
    // We can assume that outgoing ray dir actually intersects the light source.

    // Some useful quantities
    glm::vec3 light_normal         = si.normal;
    glm::vec3 light_ray_dir        = glm::normalize(si.position - last_si->position);
    float light_ray_length_squared = glm::length2(si.position - last_si->position);

    // The probability of sampling any position on the surface of the mesh is the reciprocal of its surface area.
    float light_position_pdf = 1 / total_area;

    // Probability of sampling this direction via light source sampling
    float cos_theta_on_light  = glm::abs(glm::dot(light_ray_dir, light_normal));
    float light_direction_pdf = light_position_pdf * light_ray_length_squared / cos_theta_on_light;

    return light_direction_pdf;
}
}    // namespace detail

extern "C" __device__ atcg::EmitterSamplingResult
__direct_callable__sample_meshemitter(const atcg::AnyInteraction& si,
                                      const atcg::SampledWavelengths& wavelengths,
                                      atcg::PCG32& rng)
{
    const atcg::MeshEmitterData* sbt_data = *reinterpret_cast<const atcg::MeshEmitterData**>(optixGetSbtDataPointer());
    atcg::EmitterSamplingResult result    = detail::sampleMeshEmitter(si,
                                                                   sbt_data->mesh_cdf,
                                                                   sbt_data->positions,
                                                                   sbt_data->uvs,
                                                                   sbt_data->faces,
                                                                   sbt_data->num_faces,
                                                                   sbt_data->total_area,
                                                                   sbt_data->local_to_world,
                                                                   sbt_data->world_to_local,
                                                                   rng);

    glm::vec3 emissive_color = sbt_data->emissive_texture.read(glm::vec2(result.uvs));

    result.radiance_weight_at_receiver =
        atcg::SampledSpectrum::fromRGB(sbt_data->emitter_scaling * emissive_color, wavelengths) / result.sampling_pdf;

    return result;
}

extern "C" __device__ atcg::SampledSpectrum
__direct_callable__eval_meshemitter(const atcg::SurfaceInteraction& si, const atcg::SampledWavelengths& wavelengths)
{
    const atcg::MeshEmitterData* sbt_data = *reinterpret_cast<const atcg::MeshEmitterData**>(optixGetSbtDataPointer());

    glm::vec3 emissive_color = sbt_data->emissive_texture.read(si.uv);

    return atcg::SampledSpectrum::fromRGB(sbt_data->emitter_scaling * emissive_color, wavelengths);
}

extern "C" __device__ CuDiff::Dual<6, glm::vec3>
__direct_callable__eval_forward_meshemitter(const atcg::DualSurfaceInteraction& si,
                                            const atcg::SampledWavelengths& wavelengths)
{
    const atcg::MeshEmitterData* sbt_data = *reinterpret_cast<const atcg::MeshEmitterData**>(optixGetSbtDataPointer());

    CuDiff::Dual<6, glm::vec3> emissive_color = sbt_data->emissive_texture.read(si.uv);

    return emissive_color * sbt_data->emitter_scaling;
}

extern "C" __device__ atcg::DualEmitterSamplingResult
__direct_callable__sample_forward_meshemitter(const atcg::DualSurfaceInteraction& si,
                                              const atcg::SampledWavelengths& wavelengths,
                                              atcg::PCG32& rng)
{
    const atcg::MeshEmitterData* sbt_data = *reinterpret_cast<const atcg::MeshEmitterData**>(optixGetSbtDataPointer());

    atcg::DualEmitterSamplingResult result;

    const float* mesh_cdf          = sbt_data->mesh_cdf;
    const glm::vec3* positions     = sbt_data->positions;
    const glm::vec3* uvs           = sbt_data->uvs;
    const glm::u32vec3* faces      = sbt_data->faces;
    const uint32_t num_faces       = sbt_data->num_faces;
    const float total_area         = sbt_data->total_area;
    const glm::mat4 local_to_world = sbt_data->local_to_world;
    const glm::mat4 world_to_local = sbt_data->world_to_local;

    // Select the triangle to sample a direction from uniformly at random, proportional to its surface area
    uint32_t triangle_index = 0;
    // Sample the barycentric coordinates on the triangle uniformly.
    glm::vec2 triangle_barys = glm::vec2(0, 0);

    triangle_index = atcg::Math::binary_search(mesh_cdf, rng.next1d(), num_faces);

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
    glm::vec2 uv = (1.0f - triangle_barys.x - triangle_barys.y) * UV0 + triangle_barys.x * UV1 + triangle_barys.y * UV2;

    // Compute local normal
    glm::vec3 local_light_normal = glm::cross(P1 - P0, P2 - P0);
    // Normals are transformed by (A^-1)^T instead of A
    glm::vec3 light_normal = glm::normalize(glm::transpose(glm::mat3(world_to_local)) * local_light_normal);

    // Assemble sampling result
    result.sampling_pdf = 0;    // initialize with invalid sample

    // light source sampling
    result.direction_to_light = CuDiff::normalize(light_position - si.position);
    auto light_distance       = CuDiff::length(light_position - si.position) + 1e-5f;
    result.distance_to_light  = light_distance.val();

    float one_over_light_position_pdf = total_area;
    auto cos_theta_on_light           = CuDiff::abs(CuDiff::dot(result.direction_to_light, light_normal));
    auto one_over_light_direction_pdf =
        one_over_light_position_pdf * cos_theta_on_light / (light_distance * light_distance);


    // Probability of sampling this direction via light source sampling
    result.sampling_pdf = 1 / (one_over_light_direction_pdf + 1e-5f);

    auto emissive_color = sbt_data->emissive_texture.read(uv);

    result.radiance_weight_at_receiver = sbt_data->emitter_scaling * emissive_color * one_over_light_direction_pdf;

    return result;
}

extern "C" __device__ float __direct_callable__evalpdf_meshemitter(const atcg::AnyInteraction& last_si,
                                                                   const atcg::SurfaceInteraction& si)
{
    const atcg::MeshEmitterData* sbt_data = *reinterpret_cast<const atcg::MeshEmitterData**>(optixGetSbtDataPointer());
    // We can assume that outgoing ray dir actually intersects the light source.

    return detail::evalMeshEmitterPDF(last_si, sbt_data->total_area, si);
}

extern "C" __device__ atcg::EdgeSamplingResult
__direct_callable__sample_edge_meshemitter(const atcg::SurfaceInteraction& si,
                                           const atcg::SampledWavelengths& wavelengths,
                                           atcg::PCG32& rng)
{
    const atcg::MeshEmitterData* sbt_data = *reinterpret_cast<const atcg::MeshEmitterData**>(optixGetSbtDataPointer());


    uint32_t num_edges  = sbt_data->num_edges;
    uint32_t edge_index = static_cast<uint32_t>(rng.next1d() * num_edges);
    edge_index          = glm::min(edge_index, num_edges - 1);    // TODO biased wrt edge length

    glm::ivec2 edge_vertex_indices = sbt_data->edges[edge_index];
    glm::vec3 edge_vertex_pos0 = sbt_data->local_to_world * glm::vec4(sbt_data->positions[edge_vertex_indices.x], 1.0f);
    glm::vec3 edge_vertex_pos1 = sbt_data->local_to_world * glm::vec4(sbt_data->positions[edge_vertex_indices.y], 1.0f);

    atcg::EdgeSamplingResult result;

    glm::vec3 direction = edge_vertex_pos1 - edge_vertex_pos0;
    float edge_length   = glm::length(direction);

    float dx = rng.next1d();

    result.position          = edge_vertex_pos0 + dx * direction;
    result.direction         = direction / edge_length;
    glm::vec3 w              = glm::normalize(result.position - si.position);
    result.normal            = glm::normalize(glm::cross(direction, w));    // Sign?
    result.pdf               = 1.0f / (edge_length * (float)num_edges);
    glm::vec3 emissive_color = sbt_data->emissive_texture.read(si.uv);
    result.radiance_weight   = atcg::SampledSpectrum::fromRGB(sbt_data->emitter_scaling * emissive_color, wavelengths);

    return result;
}