#include <Pathtracing/EmitterModels.h>
#include <Math/Functions.h>

#include <Pathtracing/Common.h>

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
    auto emissive_texture = material.getEmissiveTexture()->getData(atcg::GPU);

    MeshEmitterData data;

    ::detail::convertToTextureObject(emissive_texture, _emissive_texture, data.emissive_texture);

    data.emitter_scaling = material.emission_scale;

    data.positions = (glm::vec3*)positions.data_ptr();
    data.uvs       = (glm::vec3*)uvs.data_ptr();
    data.faces     = (glm::u32vec3*)faces.data_ptr();

    _mesh_cdf = torch::zeros({faces.size(0)}, atcg::TensorOptions::floatHostOptions());

    {
        detail::computeMeshTrianglePDFKernel(positions, faces, transform, _mesh_cdf);
    }

    {
        detail::computeMeshTriangleCDFKernel(_mesh_cdf);
    }

    data.total_area = _mesh_cdf.index({_mesh_cdf.size(0) - 1}).item<float>();
    {
        detail::normalizeMeshTriangleCDFKernel(_mesh_cdf, data.total_area);
    }

    data.local_to_world = transform;
    data.world_to_local = glm::inverse(transform);
    data.num_faces      = faces.size(0);
    data.mesh_cdf       = _mesh_cdf.data_ptr<float>();

    _mesh_emitter_data.upload(&data);
}

void MeshEmitter::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                     const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_emitter_filename = "./bin/EmitterModels.ptx";
    auto sample_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__sample_meshemitter"});
    auto eval_prog_group = pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__eval_meshemitter"});
    auto evalpdf_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__evalpdf_meshemitter"});
    uint32_t sample_idx   = sbt->addCallableEntry(sample_prog_group, _mesh_emitter_data.get());
    uint32_t eval_idx     = sbt->addCallableEntry(eval_prog_group, _mesh_emitter_data.get());
    uint32_t eval_pdf_idx = sbt->addCallableEntry(evalpdf_prog_group, _mesh_emitter_data.get());

    EmitterVPtrTable table;
    table.sampleCallIndex  = sample_idx;
    table.evalCallIndex    = eval_idx;
    table.evalPdfCallIndex = eval_pdf_idx;

    _vptr_table.upload(&table);
}

// TODO
MeshEmitter::~MeshEmitter() {}

EnvironmentEmitter::EnvironmentEmitter(const atcg::ref_ptr<atcg::Texture2D>& texture)
{
    auto environment_texture = texture->getData(atcg::GPU);

    EnvironmentEmitterData data;

    ::detail::convertToTextureObject(environment_texture, _environment_texture, data.environment_texture);

    _environment_emitter_data.upload(&data);
}

// TODO:
EnvironmentEmitter::~EnvironmentEmitter() {}

void EnvironmentEmitter::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                            const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_emitter_filename = "./bin/EmitterModels.ptx";
    auto sample_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__sample_environmentemitter"});
    auto eval_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__eval_environmentemitter"});
    auto evalpdf_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__evalpdf_environmentemitter"});
    uint32_t sample_idx  = sbt->addCallableEntry(sample_prog_group, _environment_emitter_data.get());
    uint32_t eval_idx    = sbt->addCallableEntry(eval_prog_group, _environment_emitter_data.get());
    uint32_t evalpdf_idx = sbt->addCallableEntry(evalpdf_prog_group, _environment_emitter_data.get());

    EmitterVPtrTable table;
    table.sampleCallIndex  = sample_idx;
    table.evalCallIndex    = eval_idx;
    table.evalPdfCallIndex = evalpdf_idx;

    _vptr_table.upload(&table);
}
}    // namespace atcg