#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>
#include <Pathtracing/SurfaceInteraction.h>

#include <Pathtracing/Emitter/EmitterModels.cuh>
#include <Pathtracing/BSDF/BSDFModels.cuh>

template<typename T>
inline __device__ uint32_t binary_search(T* sorted_array, T value, uint32_t size)
{
    // Find first element in sorted_array that is larger than value.
    uint32_t left  = 0;
    uint32_t right = size - 1;
    while(left < right)
    {
        uint32_t mid = (left + right) / 2;
        if(sorted_array[mid] < value)
            left = mid + 1;
        else
            right = mid;
    }
    return left;
}

extern "C" __device__ atcg::EmitterSamplingResult
__direct_callable__sample_meshemitter(const atcg::SurfaceInteraction& si, atcg::PCG32& rng)
{
    const atcg::MeshEmitterData* sbt_data = *reinterpret_cast<const atcg::MeshEmitterData**>(optixGetSbtDataPointer());
    atcg::EmitterSamplingResult result;

    // Select the triangle to sample a direction from uniformly at random, proportional to its surface area
    uint32_t triangle_index = 0;
    // Sample the barycentric coordinates on the triangle uniformly.
    glm::vec2 triangle_barys = glm::vec2(0, 0);

    triangle_index = binary_search(sbt_data->mesh_cdf, rng.next1d(), sbt_data->num_faces);

    triangle_barys = rng.next2d();
    // Mirror barys at diagonal line to cover a triangle instead of a square
    if(triangle_barys.x + triangle_barys.y > 1) triangle_barys = glm::vec2(1) - triangle_barys;


    // Compute the `light_position` using the triangle_index and the triangle_barys on the mesh:

    // Indices of triangle vertices in the mesh
    glm::u32vec3 vertex_indices = sbt_data->faces[triangle_index];

    // Vertex positions of selected triangle
    glm::vec3 P0 = sbt_data->positions[vertex_indices.x];
    glm::vec3 P1 = sbt_data->positions[vertex_indices.y];
    glm::vec3 P2 = sbt_data->positions[vertex_indices.z];

    // Compute local position
    glm::vec3 local_light_position =
        (1.0f - triangle_barys.x - triangle_barys.y) * P0 + triangle_barys.x * P1 + triangle_barys.y * P2;
    // Transform local position to world position
    glm::vec3 light_position = glm::vec3(sbt_data->local_to_world * glm::vec4(local_light_position, 1));

    // Compute local normal
    glm::vec3 local_light_normal = glm::cross(P1 - P0, P2 - P0);
    // Normals are transformed by (A^-1)^T instead of A
    glm::vec3 light_normal = glm::normalize(glm::transpose(glm::mat3(sbt_data->world_to_local)) * local_light_normal);

    // Assemble sampling result
    result.sampling_pdf = 0;    // initialize with invalid sample

    // light source sampling
    result.direction_to_light       = glm::normalize(light_position - si.position);
    float distance_to_light_squared = glm::length2(light_position - si.position) + 1e-5f;
    result.distance_to_light        = glm::length(light_position - si.position) + 1e-5f;
    result.normal_at_light          = light_normal;

    float one_over_light_position_pdf  = sbt_data->total_area;
    float cos_theta_on_light           = glm::abs(glm::dot(result.direction_to_light, light_normal));
    float one_over_light_direction_pdf = one_over_light_position_pdf * cos_theta_on_light / distance_to_light_squared;

    float4 color_u           = tex2D<float4>(sbt_data->emissive_texture, si.uv.x, si.uv.y);
    glm::vec3 emissive_color = glm::vec3(color_u.x, color_u.y, color_u.z);

    result.radiance_weight_at_receiver = sbt_data->emitter_scaling * emissive_color * one_over_light_direction_pdf;

    // Probability of sampling this direction via light source sampling
    result.sampling_pdf = 1 / one_over_light_direction_pdf;

    return result;
}

extern "C" __device__ glm::vec3 __direct_callable__eval_meshemitter(const atcg::SurfaceInteraction& si)
{
    const atcg::MeshEmitterData* sbt_data = *reinterpret_cast<const atcg::MeshEmitterData**>(optixGetSbtDataPointer());

    float4 color_u           = tex2D<float4>(sbt_data->emissive_texture, si.uv.x, si.uv.y);
    glm::vec3 emissive_color = glm::vec3(color_u.x, color_u.y, color_u.z);

    return emissive_color * sbt_data->emitter_scaling;
}

extern "C" __device__ float __direct_callable__evalpdf_meshemitter(const atcg::SurfaceInteraction& last_si,
                                                                   const atcg::SurfaceInteraction& si)
{
    const atcg::MeshEmitterData* sbt_data = *reinterpret_cast<const atcg::MeshEmitterData**>(optixGetSbtDataPointer());
    // We can assume that outgoing ray dir actually intersects the light source.

    // Some useful quantities
    glm::vec3 light_normal         = si.normal;
    glm::vec3 light_ray_dir        = glm::normalize(si.position - last_si.position);
    float light_ray_length_squared = glm::length2(si.position - last_si.position);

    // The probability of sampling any position on the surface of the mesh is the reciprocal of its surface area.
    float light_position_pdf = 1 / sbt_data->total_area;

    // Probability of sampling this direction via light source sampling
    float cos_theta_on_light  = glm::abs(glm::dot(light_ray_dir, light_normal));
    float light_direction_pdf = light_position_pdf * light_ray_length_squared / cos_theta_on_light;

    return light_direction_pdf;
}

extern "C" __device__ atcg::EmitterSamplingResult
__direct_callable__sample_environmentemitter(const atcg::SurfaceInteraction& si, atcg::PCG32& rng)
{
    const atcg::EnvironmentEmitterData* sbt_data =
        *reinterpret_cast<const atcg::EnvironmentEmitterData**>(optixGetSbtDataPointer());

    atcg::EmitterSamplingResult result;

    glm::vec3 random_dir = atcg::warp_square_to_hemisphere_cosine(rng.next2d());
    float pdf            = atcg::warp_square_to_hemisphere_cosine_pdf(random_dir);
    glm::mat3 frame      = atcg::compute_local_frame(si.normal);

    random_dir = frame * random_dir;

    glm::vec3 ray_dir = si.incoming_direction;

    float theta = std::acos(ray_dir.y) / glm::pi<float>();
    float phi   = (std::atan2(ray_dir.z, ray_dir.x) + glm::pi<float>()) / (2.0f * glm::pi<float>());

    glm::vec2 uv(phi, theta);

    float4 color = tex2D<float4>(sbt_data->environment_texture, uv.x, 1.0f - uv.y);

    result.distance_to_light           = std::numeric_limits<float>::infinity();
    result.sampling_pdf                = pdf;
    result.radiance_weight_at_receiver = glm::vec3(color.x, color.y, color.z) / pdf;

    return result;
}

extern "C" __device__ glm::vec3 __direct_callable__eval_environmentemitter(const atcg::SurfaceInteraction& si)
{
    const atcg::EnvironmentEmitterData* sbt_data =
        *reinterpret_cast<const atcg::EnvironmentEmitterData**>(optixGetSbtDataPointer());

    glm::vec3 ray_dir = si.incoming_direction;

    float theta = std::acos(ray_dir.y) / glm::pi<float>();
    float phi   = (std::atan2(ray_dir.z, ray_dir.x) + glm::pi<float>()) / (2.0f * glm::pi<float>());

    glm::vec2 uv(phi, theta);

    float4 color = tex2D<float4>(sbt_data->environment_texture, uv.x, 1.0f - uv.y);

    return glm::vec3(color.x, color.y, color.z);
}

extern "C" __device__ float __direct_callable__evalpdf_environmentemitter(const atcg::SurfaceInteraction& last_si,
                                                                          const atcg::SurfaceInteraction& si)
{
    const atcg::EnvironmentEmitterData* sbt_data =
        *reinterpret_cast<const atcg::EnvironmentEmitterData**>(optixGetSbtDataPointer());
    // We can assume that outgoing ray dir actually intersects the light source.

    // Probability of sampling this direction via light source sampling
    glm::mat3 local_frame     = atcg::compute_local_frame(last_si.normal);
    glm::vec3 local_direction = glm::transpose(local_frame) * si.incoming_direction;

    return atcg::warp_square_to_hemisphere_cosine_pdf(local_direction);
}
