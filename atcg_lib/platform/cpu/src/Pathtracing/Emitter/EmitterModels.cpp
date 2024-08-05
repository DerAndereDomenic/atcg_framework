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
    glm::vec3 emissive_color = texture(_emissive_texture, si.uv);

    return evalMeshEmitter(emissive_color, _emitter_scaling);
}

EmitterSamplingResult MeshEmitter::sampleLight(const SurfaceInteraction& si, PCG32& rng) const
{
    EmitterSamplingResult result = sampleMeshEmitter(si,
                                                     _mesh_cdf.data_ptr<float>(),
                                                     (glm::vec3*)_positions.data_ptr(),
                                                     (glm::vec3*)_uvs.data_ptr(),
                                                     (glm::u32vec3*)_faces.data_ptr(),
                                                     _faces.size(0),
                                                     _total_area,
                                                     _local_to_world,
                                                     _world_to_local,
                                                     rng);

    glm::vec3 emissive_color = texture(_emissive_texture, si.uv);

    result.radiance_weight_at_receiver = _emitter_scaling * emissive_color / result.sampling_pdf;

    return result;
}

float MeshEmitter::evalLightSamplingPdf(const SurfaceInteraction& last_si, const SurfaceInteraction& si) const
{
    // We can assume that outgoing ray dir actually intersects the light source.

    return evalMeshEmitterPDF(last_si, _total_area, si);
}

EnvironmentEmitter::EnvironmentEmitter(const atcg::ref_ptr<atcg::Texture2D>& environment_texture)
{
    _environment_texture = environment_texture->getData(atcg::CPU);
}

EnvironmentEmitter::~EnvironmentEmitter() {}

glm::vec3 EnvironmentEmitter::evalLight(const SurfaceInteraction& si) const
{
    auto uv = evalEnvironmentEmitter(si);

    return texture(_environment_texture, uv);
}

EmitterSamplingResult EnvironmentEmitter::sampleLight(const SurfaceInteraction& si, PCG32& rng) const
{
    auto result                        = sampleEnvironmentEmitter(si, rng);
    result.radiance_weight_at_receiver = texture(_environment_texture, result.uvs) / result.sampling_pdf;

    return result;
}

float EnvironmentEmitter::evalLightSamplingPdf(const SurfaceInteraction& last_si, const SurfaceInteraction& si) const
{
    return evalEnvironmentEmitterSamplingPdf(last_si, si);
}
}    // namespace atcg