#include <BSDF/PBRBSDF.h>

#include <Core/Common.h>
#include <BSDF/BSDFFactory.h>
#include <Renderer/Texture.h>

namespace atcg
{

PBRBSDF::PBRBSDF(const Dictionary& dict)
{
    atcg::ref_ptr<OpaqueMaterial> material =
        std::dynamic_pointer_cast<atcg::OpaqueMaterial>(dict.getValue<atcg::ref_ptr<Material>>("material"));

    _diffuse_texture   = std::dynamic_pointer_cast<Texture2D>(material->getDiffuseTexture()->clone());
    _metallic_texture  = std::dynamic_pointer_cast<Texture2D>(material->getMetallicTexture()->clone());
    _roughness_texture = std::dynamic_pointer_cast<Texture2D>(material->getRoughnessTexture()->clone());

    PBRBSDFData data;

    data.diffuse_texture.texture_data.texture   = _diffuse_texture->getTextureObject();
    data.diffuse_texture.spec                   = _diffuse_texture->getSpecification();
    data.metallic_texture.texture_data.texture  = _metallic_texture->getTextureObject();
    data.metallic_texture.spec                  = _metallic_texture->getSpecification();
    data.roughness_texture.texture_data.texture = _roughness_texture->getTextureObject();
    data.roughness_texture.spec                 = _roughness_texture->getSpecification();

    _flags = BSDFComponentType::GlossyReflection | BSDFComponentType::DiffuseReflection;

    _bsdf_data_buffer.upload(&data);
}

PBRBSDF::~PBRBSDF()
{
    _diffuse_texture->unmapDevicePointers();
    _metallic_texture->unmapDevicePointers();
    _roughness_texture->unmapDevicePointers();
}

void PBRBSDF::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                 const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_bsdf_filename = "./bin/PBRBSDF_ptx.ptx";
    auto sample_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_pbrbsdf"});
    auto eval_prog_group   = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_pbrbsdf"});
    uint32_t sample_idx    = sbt->addCallableEntry(sample_prog_group, _bsdf_data_buffer.get());
    uint32_t eval_idx      = sbt->addCallableEntry(eval_prog_group, _bsdf_data_buffer.get());

    BSDFVPtrTable table;
    table.sampleCallIndex = sample_idx;
    table.evalCallIndex   = eval_idx;
    table.flags           = _flags;

    _vptr_table.upload(&table);

    markInitialized();
}

ATCG_REGISTER_BSDF("Opaque", PBRBSDF);
}    // namespace atcg