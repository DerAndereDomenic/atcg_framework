#include <Renderer/Material.h>
#include <Renderer/Renderer.h>
#include <Renderer/Shader.h>
#include <Core/Assert.h>

namespace atcg
{
Material::Material(MaterialType type) : _material_type(type)
{
    TextureSpecification spec_diffuse;
    spec_diffuse.width  = 1;
    spec_diffuse.height = 1;
    glm::u8vec4 white(255);
    _diffuse_texture = atcg::Texture2D::create(&white, spec_diffuse);

    TextureSpecification spec_normal;
    spec_normal.width  = 1;
    spec_normal.height = 1;
    glm::u8vec4 normal(127, 127, 255, 255);
    _normal_texture = atcg::Texture2D::create(&normal, spec_normal);

    TextureSpecification spec_roughness;
    spec_roughness.width  = 1;
    spec_roughness.height = 1;
    spec_roughness.format = TextureFormat::RFLOAT;
    float roughness       = 1.0f;
    _roughness_texture    = atcg::Texture2D::create(&roughness, spec_roughness);

    TextureSpecification spec_metallic;
    spec_metallic.width  = 1;
    spec_metallic.height = 1;
    spec_metallic.format = TextureFormat::RFLOAT;
    float metallic       = 0.0f;
    _metallic_texture    = atcg::Texture2D::create(&metallic, spec_metallic);

    TextureSpecification spec_ior;
    spec_ior.width  = 1;
    spec_ior.height = 1;
    spec_ior.format = TextureFormat::RFLOAT;
    float ior_value = 1.45f;
    _ior_texture    = atcg::Texture2D::create(&ior_value, spec_ior);
}

void Material::setDiffuseColor(const glm::vec4& color)
{
    TextureSpecification spec_diffuse;
    spec_diffuse.width  = 1;
    spec_diffuse.height = 1;
    glm::u8vec4 color_quant((uint8_t)(color[0] * 255.0f),
                            (uint8_t)(color[1] * 255.0f),
                            (uint8_t)(color[2] * 255.0f),
                            (uint8_t)(color[3] * 255.0f));
    _diffuse_texture = atcg::Texture2D::create(&color_quant, spec_diffuse);
}

void Material::setDiffuseColor(const glm::vec3& color)
{
    setDiffuseColor(glm::vec4(color, 1.0f));
}

void Material::setRoughness(const float roughness)
{
    TextureSpecification spec_roughness;
    spec_roughness.width  = 1;
    spec_roughness.height = 1;
    spec_roughness.format = TextureFormat::RFLOAT;
    _roughness_texture    = atcg::Texture2D::create(&roughness, spec_roughness);
}

void Material::setMetallic(const float metallic)
{
    TextureSpecification spec_metallic;
    spec_metallic.width  = 1;
    spec_metallic.height = 1;
    spec_metallic.format = TextureFormat::RFLOAT;
    _metallic_texture    = atcg::Texture2D::create(&metallic, spec_metallic);
}

void Material::setIor(const float ior_value)
{
    TextureSpecification spec_ior;
    spec_ior.width  = 1;
    spec_ior.height = 1;
    spec_ior.format = TextureFormat::RFLOAT;
    _ior_texture    = atcg::Texture2D::create(&ior_value, spec_ior);
}

void Material::removeNormalMap()
{
    TextureSpecification spec_normal;
    spec_normal.width  = 1;
    spec_normal.height = 1;
    glm::u8vec4 normal(127, 127, 255, 255);
    _normal_texture = atcg::Texture2D::create(&normal, spec_normal);
}

void Material::uploadMaterial(RendererSystem* renderer, const atcg::ref_ptr<Shader>& shader)
{
    ATCG_ASSERT(!_uploaded, "Material was already uploaded");

    uint32_t diffuse_id = renderer->popTextureID();
    GraphicsCommand::bindTexture(diffuse_id, getDiffuseTexture());
    shader->setInt("texture_diffuse", diffuse_id);
    _used_texture_ids[0] = diffuse_id;

    uint32_t normal_id = renderer->popTextureID();
    GraphicsCommand::bindTexture(normal_id, getNormalTexture());
    shader->setInt("texture_normal", normal_id);
    _used_texture_ids[1] = normal_id;

    uint32_t roughness_id = renderer->popTextureID();
    GraphicsCommand::bindTexture(roughness_id, getRoughnessTexture());
    shader->setInt("texture_roughness", roughness_id);
    _used_texture_ids[2] = roughness_id;

    uint32_t metallic_id = renderer->popTextureID();
    GraphicsCommand::bindTexture(metallic_id, getMetallicTexture());
    shader->setInt("texture_metallic", metallic_id);
    _used_texture_ids[3] = metallic_id;

    uint32_t ior_id = renderer->popTextureID();
    GraphicsCommand::bindTexture(ior_id, getIorTexture());
    shader->setInt("texture_ior", ior_id);
    _used_texture_ids[4] = ior_id;

    switch(_material_type)
    {
        case MaterialType::MATERIAL_TYPE_OPAQUE:
        {
            shader->selectSubroutine("sr_eval_brdf", "eval_brdf_pbr");
            shader->selectSubroutine("sr_image_based_lighting", "image_based_lighting_pbr");
        }
        break;
        case MaterialType::MATERIAL_TYPE_GLASS:
        {
            shader->selectSubroutine("sr_eval_brdf", "eval_brdf_glass");
            shader->selectSubroutine("sr_image_based_lighting", "image_based_lighting_glass");
        }
        break;
        case MaterialType::MATERIAL_TYPE_NULL:
        {
            shader->selectSubroutine("sr_eval_brdf", "eval_brdf_null");
            shader->selectSubroutine("sr_image_based_lighting", "image_based_lighting_null");
        }
        break;
    }

    _uploaded = true;
}

void Material::releaseTextureIDs(RendererSystem* renderer)
{
    ATCG_ASSERT(_uploaded, "Tried freeing material ids without while material is not uploaded");

    renderer->pushTextureID(_used_texture_ids[0]);
    renderer->pushTextureID(_used_texture_ids[1]);
    renderer->pushTextureID(_used_texture_ids[2]);
    renderer->pushTextureID(_used_texture_ids[3]);
    renderer->pushTextureID(_used_texture_ids[4]);

    _uploaded = false;
}

}    // namespace atcg