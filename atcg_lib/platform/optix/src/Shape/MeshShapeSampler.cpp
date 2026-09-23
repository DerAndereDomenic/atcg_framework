#include <Shape/MeshShapeSampler.h>
#include <Shape/MeshShape.h>
#include <Shape/MeshKernels.h>

namespace atcg
{
MeshShapeSampler::MeshShapeSampler(const atcg::Dictionary& dict) : ShapeSampler(dict)
{
    MeshSamplerData data;

    auto mesh_shape = std::dynamic_pointer_cast<MeshShape>(_shape);
    if(!mesh_shape)
    {
        ATCG_ERROR("Tried to instantiate a MeshShapeSampler with a non-MeshShape shape");
        return;
    }

    torch::Tensor positions = mesh_shape->getPositions();
    torch::Tensor uvs       = mesh_shape->getUVs();
    torch::Tensor faces     = mesh_shape->getFaces();

    data.positions = (glm::vec3*)positions.data_ptr();
    data.uvs       = (glm::vec3*)uvs.data_ptr();
    data.faces     = (glm::u32vec3*)faces.data_ptr();
    data.num_faces = faces.size(0);

    _cdf = torch::zeros({faces.size(0)}, atcg::TensorOptions::floatDeviceOptions());

    auto device = _cdf.device();

    computeMeshTrianglePDFKernel(positions, faces, _transform, _cdf);
    computeMeshTriangleCDFKernel(_cdf);

    data.total_area = _cdf.index({_cdf.size(0) - 1}).cpu().item<float>();
    normalizeMeshTriangleCDFKernel(_cdf, data.total_area);

    data.mesh_cdf       = (float*)_cdf.data_ptr();
    data.local_to_world = _transform;
    data.world_to_local = glm::inverse(_transform);

    _data.upload(&data);
}

MeshShapeSampler::~MeshShapeSampler() {}

void MeshShapeSampler::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                          const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_bsdf_filename = "./bin/MeshSampler_ptx.ptx";
    auto sample_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_point_mesh"});
    auto eval_prog_group   = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_edge_mesh"});
    uint32_t sample_point_idx = sbt->addCallableEntry(sample_prog_group, _data.get());
    uint32_t sample_edge_idx  = sbt->addCallableEntry(eval_prog_group, _data.get());

    ShapeSamplerVPtrTable table;
    table.sampleShapeCallIndex = sample_point_idx;
    table.sampleEdgeCallIndex  = sample_edge_idx;

    _vptr_table.upload(&table);

    markInitialized();
}
}    // namespace atcg