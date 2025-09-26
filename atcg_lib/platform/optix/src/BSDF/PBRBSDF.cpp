#include <BSDF/PBRBSDF.h>

#include <Core/Common.h>
#include <BSDF/BSDFFactory.h>
#include <Renderer/Texture.h>

namespace atcg
{

PBRBSDF::PBRBSDF(const Dictionary& dict)
{
    atcg::ref_ptr<Material> material = dict.getValue<atcg::ref_ptr<Material>>("material");

    _diffuse_texture   = std::dynamic_pointer_cast<Texture2D>(material->getDiffuseTexture()->clone());
    _metallic_texture  = std::dynamic_pointer_cast<Texture2D>(material->getMetallicTexture()->clone());
    _roughness_texture = std::dynamic_pointer_cast<Texture2D>(material->getRoughnessTexture()->clone());

    // Cheap way to create a tensor that exactly matches the texture format
    _grad_diffuse =
        torch::zeros({_diffuse_texture->height(), _diffuse_texture->width(), 3}, TensorOptions::floatDeviceOptions());
    _grad_metallic =
        torch::zeros({_metallic_texture->height(), _metallic_texture->width()}, TensorOptions::floatDeviceOptions());
    _grad_roughness =
        torch::zeros({_roughness_texture->height(), _roughness_texture->width()}, TensorOptions::floatDeviceOptions());

    PBRBSDFData data;

    data.diffuse_texture.texture_data.texture   = _diffuse_texture->getTextureObject();
    data.diffuse_texture.spec                   = _diffuse_texture->getSpecification();
    data.metallic_texture.texture_data.texture  = _metallic_texture->getTextureObject();
    data.metallic_texture.spec                  = _metallic_texture->getSpecification();
    data.roughness_texture.texture_data.texture = _roughness_texture->getTextureObject();
    data.roughness_texture.spec                 = _roughness_texture->getSpecification();

    data.grad_diffuse   = (glm::vec3*)_grad_diffuse.data_ptr();
    data.grad_roughness = (float*)_grad_roughness.data_ptr();
    data.grad_metallic  = (float*)_grad_metallic.data_ptr();

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
    auto sample_prog_group   = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_pbrbsdf"});
    auto eval_prog_group     = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_pbrbsdf"});
    auto backward_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__grad_pbrbsdf"});
    uint32_t sample_idx      = sbt->addCallableEntry(sample_prog_group, _bsdf_data_buffer.get());
    uint32_t eval_idx        = sbt->addCallableEntry(eval_prog_group, _bsdf_data_buffer.get());
    uint32_t backward_idx    = sbt->addCallableEntry(backward_prog_group, _bsdf_data_buffer.get());

    BSDFVPtrTable table;
    table.sampleCallIndex   = sample_idx;
    table.evalCallIndex     = eval_idx;
    table.evalBackwardIndex = backward_idx;
    table.flags             = _flags;

    _vptr_table.upload(&table);
}

void PBRBSDF::zero_grad()
{
    _grad_diffuse.zero_();
    _grad_metallic.zero_();
    _grad_roughness.zero_();
}

void PBRBSDF::update(float lr)
{
    auto data           = _diffuse_texture->getData(atcg::GPU);
    auto orig_type      = data.scalar_type();
    bool transform_back = false;
    if(!data.is_floating_point())
    {
        data           = data.to(torch::kFloat32) / 255.0f;
        transform_back = true;
    }

    std::cout << _grad_diffuse << "\n";

    // data.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
    //                 data.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3)}) -
    //                     lr * _grad_diffuse);

    if(transform_back)
    {
        data = (data * 255.0f).to(orig_type);
    }

    _diffuse_texture->setData(data);
}

ATCG_REGISTER_BSDF(MaterialType::MATERIAL_TYPE_OPAQUE, PBRBSDF);
}    // namespace atcg