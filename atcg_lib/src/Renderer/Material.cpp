#include <Renderer/Material.h>

namespace atcg
{
Material::Material()
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

void Material::removeNormalMap()
{
    TextureSpecification spec_normal;
    spec_normal.width  = 1;
    spec_normal.height = 1;
    glm::u8vec4 normal(127, 127, 255, 255);
    _normal_texture = atcg::Texture2D::create(&normal, spec_normal);
}

}    // namespace atcg