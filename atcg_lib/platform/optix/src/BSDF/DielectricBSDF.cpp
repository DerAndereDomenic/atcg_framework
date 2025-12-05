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
    _ior_texture       = std::dynamic_pointer_cast<Texture2D>(material->getIorTexture()->clone());

    DielectricBSDFData data;

    data.diffuse_texture.texture_data.texture   = _diffuse_texture->getTextureObject();
    data.diffuse_texture.spec                   = _diffuse_texture->getSpecification();
    data.roughness_texture.texture_data.texture = _roughness_texture->getTextureObject();
    data.roughness_texture.spec                 = _roughness_texture->getSpecification();
    data.ior_texture.texture_data.texture       = _ior_texture->getTextureObject();
    data.ior_texture.spec                       = _ior_texture->getSpecification();

    _flags = BSDFComponentType::IdealReflection | BSDFComponentType::IdealReflection;

    _bsdf_data_buffer.upload(&data);
}

DielectricBSDF::~DielectricBSDF()
{
    _diffuse_texture->unmapDevicePointers();
    _roughness_texture->unmapDevicePointers();
}

void PipelineInitializer<DielectricBSDF>::apply(const atcg::ref_ptr<DielectricBSDF>& component) const
{
    const std::string ptx_bsdf_filename = "./bin/DielectricBSDF_ptx.ptx";
    auto sample_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_dielectricbsdf"});
    auto eval_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_dielectricbsdf"});
    uint32_t sample_idx  = sbt->addCallableEntry(sample_prog_group, component->getDataBuffer().get());
    uint32_t eval_idx    = sbt->addCallableEntry(eval_prog_group, component->getDataBuffer().get());

    BSDFVPtrTable table;
    table.sampleCallIndex = sample_idx;
    table.evalCallIndex   = eval_idx;
    table.flags           = component->flags();

    component->getVPtrTableHolder().upload(&table);

    component->markInitialized();
}

ATCG_REGISTER_BSDF(MaterialType::MATERIAL_TYPE_GLASS, DielectricBSDF);
}    // namespace atcg