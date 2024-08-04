#include <Pathtracing/Emitter/EmitterModels.h>
#include <Math/Functions.h>

namespace atcg
{

namespace detail
{
void computeMeshTrianglePDFKernel(const torch::Tensor& positions,
                                  const torch::Tensor& indices,
                                  const glm::mat4& transform,
                                  torch::Tensor& pdf)
{
    for(int tid = 0; tid < indices.size(0); ++tid)
    {
        glm::u32vec3 triangle_indices = glm::u32vec3(indices[tid][0].item<int32_t>(),
                                                     indices[tid][1].item<int32_t>(),
                                                     indices[tid][2].item<int32_t>());
        glm::vec3 local_P0            = glm::vec3(positions[triangle_indices.x][0].item<float>(),
                                       positions[triangle_indices.x][1].item<float>(),
                                       positions[triangle_indices.x][2].item<float>());
        glm::vec3 local_P1            = glm::vec3(positions[triangle_indices.y][0].item<float>(),
                                       positions[triangle_indices.y][1].item<float>(),
                                       positions[triangle_indices.y][2].item<float>());
        glm::vec3 local_P2            = glm::vec3(positions[triangle_indices.z][0].item<float>(),
                                       positions[triangle_indices.z][1].item<float>(),
                                       positions[triangle_indices.z][2].item<float>());

        glm::vec3 P0 = glm::vec3(transform * glm::vec4(local_P0, 1));
        glm::vec3 P1 = glm::vec3(transform * glm::vec4(local_P1, 1));
        glm::vec3 P2 = glm::vec3(transform * glm::vec4(local_P2, 1));

        // Compute triangle area
        float parallelogram_area = glm::length(glm::cross(P1 - P0, P2 - P0));
        float triangle_area      = 0.5f * parallelogram_area;

        // Write unnormalized pdf
        pdf[tid] = triangle_area;
    }
}

void computeMeshTriangleCDFKernel(torch::Tensor& cdf)
{
    float acc = 0;
    for(uint32_t i = 0; i < cdf.size(0); ++i)
    {
        acc += cdf[i].item<float>();
        cdf[i] = acc;
    }
}

void normalizeMeshTriangleCDFKernel(torch::Tensor& cdf, float total_value)
{
    for(auto tid = 0; tid < cdf.size(0); ++tid)
    {
        cdf[tid] /= total_value;
    }
}

template<typename T>
inline uint32_t binary_search(T* sorted_array, T value, uint32_t size)
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

static glm::vec3 read_image(const torch::Tensor& image, const glm::vec2& uv)
{
    glm::vec4 result(0);

    uint32_t image_width  = image.size(1);
    uint32_t image_height = image.size(0);
    glm::ivec2 pixel(uv.x * image_width, image_height - uv.y * image_height);
    pixel = glm::clamp(pixel, glm::ivec2(0), glm::ivec2(image_width - 1, image_height - 1));

    glm::vec3 color;

    uint32_t channels = image.size(2);
    bool is_hdr       = image.scalar_type() == torch::kFloat32;
    if(is_hdr)
    {
        if(channels == 4 || channels == 3)
        {
            color = glm::vec3(image.index({pixel.y, pixel.x, 0}).item<float>(),
                              image.index({pixel.y, pixel.x, 1}).item<float>(),
                              image.index({pixel.y, pixel.x, 2}).item<float>());
        }
        else
        {
            color = glm::vec3(image.index({pixel.y, pixel.x, 0}).item<float>());
        }
    }
    else
    {
        if(channels == 4 || channels == 3)
        {
            glm::u8vec3 val = glm::u8vec3(image.index({pixel.y, pixel.x, 0}).item<uint8_t>(),
                                          image.index({pixel.y, pixel.x, 1}).item<uint8_t>(),
                                          image.index({pixel.y, pixel.x, 2}).item<uint8_t>());
            color           = glm::vec3((float)val.x, (float)val.y, (float)val.z) / 255.0f;
        }
        else
        {
            uint8_t val = image.index({pixel.y, pixel.x, 0}).item<uint8_t>();
            color       = glm::vec3((float)val) / 255.0f;
        }
    }

    return color;
}
}    // namespace detail

MeshEmitter::MeshEmitter(const torch::Tensor& positions,
                         const torch::Tensor& uvs,
                         const torch::Tensor& faces,
                         const glm::mat4& transform,
                         const Material& material)
{
    _emissive_texture = material.getEmissiveTexture()->getData(atcg::CPU);

    _emitter_scaling = material.emission_scale;

    _positions = positions.clone();
    _uvs       = uvs.clone();
    _faces     = faces.clone();

    _mesh_cdf = torch::zeros({faces.size(0)}, atcg::TensorOptions::floatHostOptions());

    {
        detail::computeMeshTrianglePDFKernel(positions, faces, transform, _mesh_cdf);
    }

    {
        detail::computeMeshTriangleCDFKernel(_mesh_cdf);
    }

    _total_area = _mesh_cdf.index({_mesh_cdf.size(0) - 1}).item<float>();
    {
        detail::normalizeMeshTriangleCDFKernel(_mesh_cdf, _total_area);
    }

    _local_to_world = transform;
    _world_to_local = glm::inverse(transform);
}

MeshEmitter::~MeshEmitter() {}

glm::vec3 MeshEmitter::evalLight(const SurfaceInteraction& si) const
{
    glm::vec3 emissive_color = detail::read_image(_emissive_texture, si.uv);

    return emissive_color * _emitter_scaling;
}

EmitterSamplingResult MeshEmitter::sampleLight(const SurfaceInteraction& si, PCG32& rng) const
{
    atcg::EmitterSamplingResult result;

    // Select the triangle to sample a direction from uniformly at random, proportional to its surface area
    uint32_t triangle_index = 0;
    // Sample the barycentric coordinates on the triangle uniformly.
    glm::vec2 triangle_barys = glm::vec2(0, 0);

    triangle_index = detail::binary_search(_mesh_cdf.data_ptr<float>(), rng.next1d(), _faces.size(0));

    triangle_barys = rng.next2d();
    // Mirror barys at diagonal line to cover a triangle instead of a square
    if(triangle_barys.x + triangle_barys.y > 1) triangle_barys = glm::vec2(1) - triangle_barys;


    // Compute the `light_position` using the triangle_index and the triangle_barys on the mesh:

    // Indices of triangle vertices in the mesh
    glm::u32vec3 vertex_indices = glm::u32vec3(_faces[triangle_index][0].item<int32_t>(),
                                               _faces[triangle_index][1].item<int32_t>(),
                                               _faces[triangle_index][2].item<int32_t>());

    // Vertex positions of selected triangle
    glm::vec3 P0 = glm::vec3(_positions[vertex_indices.x][0].item<float>(),
                             _positions[vertex_indices.x][1].item<float>(),
                             _positions[vertex_indices.x][2].item<float>());
    glm::vec3 P1 = glm::vec3(_positions[vertex_indices.y][0].item<float>(),
                             _positions[vertex_indices.y][1].item<float>(),
                             _positions[vertex_indices.y][2].item<float>());
    glm::vec3 P2 = glm::vec3(_positions[vertex_indices.z][0].item<float>(),
                             _positions[vertex_indices.z][1].item<float>(),
                             _positions[vertex_indices.z][2].item<float>());

    glm::vec3 UV0 = glm::vec3(_uvs[vertex_indices.x][0].item<float>(),
                              _uvs[vertex_indices.x][1].item<float>(),
                              _uvs[vertex_indices.x][2].item<float>());
    glm::vec3 UV1 = glm::vec3(_uvs[vertex_indices.y][0].item<float>(),
                              _uvs[vertex_indices.y][1].item<float>(),
                              _uvs[vertex_indices.y][2].item<float>());
    glm::vec3 UV2 = glm::vec3(_uvs[vertex_indices.z][0].item<float>(),
                              _uvs[vertex_indices.z][1].item<float>(),
                              _uvs[vertex_indices.z][2].item<float>());

    // Compute local position
    glm::vec3 local_light_position =
        (1.0f - triangle_barys.x - triangle_barys.y) * P0 + triangle_barys.x * P1 + triangle_barys.y * P2;
    // Transform local position to world position
    glm::vec3 light_position = glm::vec3(_local_to_world * glm::vec4(local_light_position, 1));

    // Compute UVS
    glm::vec3 uvs =
        (1.0f - triangle_barys.x - triangle_barys.y) * UV0 + triangle_barys.x * UV1 + triangle_barys.y * UV2;

    // Compute local normal
    glm::vec3 local_light_normal = glm::cross(P1 - P0, P2 - P0);
    // Normals are transformed by (A^-1)^T instead of A
    glm::vec3 light_normal = glm::normalize(glm::transpose(glm::mat3(_world_to_local)) * local_light_normal);

    // Assemble sampling result
    result.sampling_pdf = 0;    // initialize with invalid sample

    // light source sampling
    result.direction_to_light       = glm::normalize(light_position - si.position);
    float distance_to_light_squared = glm::length2(light_position - si.position) + 1e-5f;
    result.distance_to_light        = glm::length(light_position - si.position) + 1e-5f;
    result.normal_at_light          = light_normal;

    float one_over_light_position_pdf  = _total_area;
    float cos_theta_on_light           = glm::abs(glm::dot(result.direction_to_light, light_normal));
    float one_over_light_direction_pdf = one_over_light_position_pdf * cos_theta_on_light / distance_to_light_squared;

    glm::vec3 emissive_color = detail::read_image(_emissive_texture, si.uv);

    result.radiance_weight_at_receiver = _emitter_scaling * emissive_color * one_over_light_direction_pdf;

    // Probability of sampling this direction via light source sampling
    result.sampling_pdf = 1 / one_over_light_direction_pdf;

    return result;
}

PhotonSamplingResult MeshEmitter::samplePhoton(PCG32& rng) const
{
    atcg::PhotonSamplingResult result;

    // Select the triangle to sample a direction from uniformly at random, proportional to its surface area
    uint32_t triangle_index = 0;
    // Sample the barycentric coordinates on the triangle uniformly.
    glm::vec2 triangle_barys = glm::vec2(0, 0);

    triangle_index = detail::binary_search(_mesh_cdf.data_ptr<float>(), rng.next1d(), _faces.size(0));

    triangle_barys = rng.next2d();
    // Mirror barys at diagonal line to cover a triangle instead of a square
    if(triangle_barys.x + triangle_barys.y > 1) triangle_barys = glm::vec2(1) - triangle_barys;


    // Compute the `light_position` using the triangle_index and the triangle_barys on the mesh:

    // Indices of triangle vertices in the mesh
    glm::u32vec3 vertex_indices = glm::u32vec3(_faces[triangle_index][0].item<int32_t>(),
                                               _faces[triangle_index][1].item<int32_t>(),
                                               _faces[triangle_index][2].item<int32_t>());

    // Vertex positions of selected triangle
    glm::vec3 P0 = glm::vec3(_positions[vertex_indices.x][0].item<float>(),
                             _positions[vertex_indices.x][1].item<float>(),
                             _positions[vertex_indices.x][2].item<float>());
    glm::vec3 P1 = glm::vec3(_positions[vertex_indices.y][0].item<float>(),
                             _positions[vertex_indices.y][1].item<float>(),
                             _positions[vertex_indices.y][2].item<float>());
    glm::vec3 P2 = glm::vec3(_positions[vertex_indices.z][0].item<float>(),
                             _positions[vertex_indices.z][1].item<float>(),
                             _positions[vertex_indices.z][2].item<float>());

    glm::vec3 UV0 = glm::vec3(_uvs[vertex_indices.x][0].item<float>(),
                              _uvs[vertex_indices.x][1].item<float>(),
                              _uvs[vertex_indices.x][2].item<float>());
    glm::vec3 UV1 = glm::vec3(_uvs[vertex_indices.y][0].item<float>(),
                              _uvs[vertex_indices.y][1].item<float>(),
                              _uvs[vertex_indices.y][2].item<float>());
    glm::vec3 UV2 = glm::vec3(_uvs[vertex_indices.z][0].item<float>(),
                              _uvs[vertex_indices.z][1].item<float>(),
                              _uvs[vertex_indices.z][2].item<float>());

    // Compute local position
    glm::vec3 local_light_position =
        (1.0f - triangle_barys.x - triangle_barys.y) * P0 + triangle_barys.x * P1 + triangle_barys.y * P2;
    // Transform local position to world position
    glm::vec3 light_position = glm::vec3(_local_to_world * glm::vec4(local_light_position, 1));

    // Compute UVS
    glm::vec3 uvs =
        (1.0f - triangle_barys.x - triangle_barys.y) * UV0 + triangle_barys.x * UV1 + triangle_barys.y * UV2;

    // Compute local normal
    glm::vec3 local_light_normal = glm::cross(P1 - P0, P2 - P0);
    // Normals are transformed by (A^-1)^T instead of A
    glm::vec3 light_normal = glm::normalize(glm::transpose(glm::mat3(_world_to_local)) * local_light_normal);

    // Assemble sampling result
    result.pdf = 0;    // initialize with invalid sample

    // light source sampling

    glm::vec3 emissive_color = detail::read_image(_emissive_texture, uvs);

    glm::vec3 local_light_direction = atcg::warp_square_to_hemisphere_cosine(rng.next2d());
    float direction_pdf             = atcg::warp_square_to_hemisphere_cosine_pdf(local_light_direction);

    result.position        = light_position;
    result.direction       = atcg::Math::compute_local_frame(light_normal) * local_light_direction;
    result.pdf             = 1.0f / (_total_area)*direction_pdf;
    result.radiance_weight = emissive_color * _emitter_scaling * (_total_area);
    result.normal          = light_normal;

    return result;
}

float MeshEmitter::evalLightSamplingPdf(const SurfaceInteraction& last_si, const SurfaceInteraction& si) const
{
    // We can assume that outgoing ray dir actually intersects the light source.

    // Some useful quantities
    glm::vec3 light_normal         = si.normal;
    glm::vec3 light_ray_dir        = glm::normalize(si.position - last_si.position);
    float light_ray_length_squared = glm::length2(si.position - last_si.position);

    // The probability of sampling any position on the surface of the mesh is the reciprocal of its surface area.
    float light_position_pdf = 1 / _total_area;

    // Probability of sampling this direction via light source sampling
    float cos_theta_on_light  = glm::abs(glm::dot(light_ray_dir, light_normal));
    float light_direction_pdf = light_position_pdf * light_ray_length_squared / cos_theta_on_light;

    return light_direction_pdf;
}

EnvironmentEmitter::EnvironmentEmitter(const atcg::ref_ptr<atcg::Texture2D>& environment_texture)
{
    _environment_texture = environment_texture->getData(atcg::CPU);
}

EnvironmentEmitter::~EnvironmentEmitter() {}

glm::vec3 EnvironmentEmitter::evalLight(const SurfaceInteraction& si) const
{
    glm::vec3 ray_dir = si.incoming_direction;

    float theta = std::acos(ray_dir.y) / glm::pi<float>();
    float phi   = (std::atan2(ray_dir.z, ray_dir.x) + glm::pi<float>()) / (2.0f * glm::pi<float>());

    glm::vec2 uv(phi, theta);

    return detail::read_image(_environment_texture, uv);
}

EmitterSamplingResult EnvironmentEmitter::sampleLight(const SurfaceInteraction& si, PCG32& rng) const
{
    atcg::EmitterSamplingResult result;

    glm::vec3 random_dir = atcg::warp_square_to_hemisphere_cosine(rng.next2d());
    float pdf            = atcg::warp_square_to_hemisphere_cosine_pdf(random_dir);
    glm::mat3 frame      = atcg::Math::compute_local_frame(si.normal);

    random_dir = frame * random_dir;

    glm::vec3 ray_dir = si.incoming_direction;

    float theta = std::acos(ray_dir.y) / glm::pi<float>();
    float phi   = (std::atan2(ray_dir.z, ray_dir.x) + glm::pi<float>()) / (2.0f * glm::pi<float>());

    glm::vec2 uv(phi, theta);

    result.distance_to_light           = std::numeric_limits<float>::infinity();
    result.sampling_pdf                = pdf;
    result.radiance_weight_at_receiver = detail::read_image(_environment_texture, uv) / pdf;

    return result;
}

PhotonSamplingResult EnvironmentEmitter::samplePhoton(PCG32& rng) const
{
    // TODO
    return PhotonSamplingResult();
}

float EnvironmentEmitter::evalLightSamplingPdf(const SurfaceInteraction& last_si, const SurfaceInteraction& si) const
{
    // We can assume that outgoing ray dir actually intersects the light source.

    // Probability of sampling this direction via light source sampling
    glm::mat3 local_frame     = atcg::Math::compute_local_frame(last_si.normal);
    glm::vec3 local_direction = glm::transpose(local_frame) * si.incoming_direction;

    return atcg::warp_square_to_hemisphere_cosine_pdf(local_direction);
}
}    // namespace atcg