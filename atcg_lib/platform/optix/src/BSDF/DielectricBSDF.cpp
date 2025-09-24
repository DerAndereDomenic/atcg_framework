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

    _diffuse_texture   = std::dynamic_pointer_cast<Texture2D>(material->getDiffuseTexture()->clone());
    _roughness_texture = std::dynamic_pointer_cast<Texture2D>(material->getRoughnessTexture()->clone());

    DielectricBSDFData data;

    data.diffuse_texture   = _diffuse_texture->getTextureObject();
    data.roughness_texture = _roughness_texture->getTextureObject();
    data.ior               = material->ior;

    _flags = BSDFComponentType::IdealReflection | BSDFComponentType::IdealReflection;

    _bsdf_data_buffer.upload(&data);
}

DielectricBSDF::~DielectricBSDF()
{
    _diffuse_texture->unmapDevicePointers();
    _roughness_texture->unmapDevicePointers();
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