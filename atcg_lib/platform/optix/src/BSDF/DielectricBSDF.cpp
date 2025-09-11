#include <BSDF/DielectricBSDF.h>

#include <Renderer/Texture.h>
#include <Renderer/Material.h>
#include <BSDF/BSDFFactory.h>

#include <Core/Common.h>

namespace atcg
{

DielectricBSDF::DielectricBSDF(const Dictionary& dict)
{
    auto material = dict.getValue<atcg::ref_ptr<Material>>("material");

    auto diffuse   = material->getDiffuseTexture()->getData(atcg::GPU);
    auto roughness = material->getRoughnessTexture()->getData(atcg::GPU);

    DielectricBSDFData data;

    atcg::convertToTextureObject(diffuse, _diffuse_texture, data.diffuse_texture);
    atcg::convertToTextureObject(roughness, _roughness_texture, data.roughness_texture);
    data.ior = material->ior;

    _flags = BSDFComponentType::IdealReflection | BSDFComponentType::IdealReflection;

    _bsdf_data_buffer.upload(&data);
}

DielectricBSDF::~DielectricBSDF()
{
    DielectricBSDFData data;

    _bsdf_data_buffer.download(&data);

    CUDA_SAFE_CALL(cudaDestroyTextureObject(data.diffuse_texture));
    CUDA_SAFE_CALL(cudaDestroyTextureObject(data.roughness_texture));

    CUDA_SAFE_CALL(cudaFreeArray(_diffuse_texture));
    CUDA_SAFE_CALL(cudaFreeArray(_roughness_texture));
}

void DielectricBSDF::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                        const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_bsdf_filename = "./bin/DielectricBSDF_ptx.ptx";
    auto sample_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_dielectricbsdf"});
    auto eval_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_dielectricbsdf"});
    uint32_t sample_idx  = sbt->addCallableEntry(sample_prog_group, _bsdf_data_buffer.get());
    uint32_t eval_idx    = sbt->addCallableEntry(eval_prog_group, _bsdf_data_buffer.get());

    BSDFVPtrTable table;
    table.sampleCallIndex = sample_idx;
    table.evalCallIndex   = eval_idx;
    table.flags           = _flags;

    _vptr_table.upload(&table);
}

ATCG_REGISTER_BSDF(MaterialType::MATERIAL_TYPE_GLASS, DielectricBSDF);
}    // namespace atcg