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
    glm::u8vec4 normal(128, 128, 255, 255);
    _normal_texture = atcg::Texture2D::create(&normal, spec_normal);

    TextureSpecification spec_roughness;
    spec_roughness.width  = 1;
    spec_roughness.height = 1;
    spec_roughness.format = TextureFormat::RINT8;
    uint8_t roughness     = 255;
    _roughness_texture    = atcg::Texture2D::create(&roughness, spec_roughness);

    TextureSpecification spec_metallic;
    spec_metallic.width  = 1;
    spec_metallic.height = 1;
    spec_metallic.format = TextureFormat::RINT8;
    uint8_t metallic     = 0;
    _metallic_texture    = atcg::Texture2D::create(&metallic, spec_metallic);
}

void Material::setDiffuseColor(const glm::vec4& color)
{
    TextureSpecification spec_diffuse;
    spec_diffuse.width  = 1;
    spec_diffuse.height = 1;
    _diffuse_texture    = atcg::Texture2D::create(&color, spec_diffuse);
}

void Material::setRoughness(const float roughness)
{
    TextureSpecification spec_roughness;
    spec_roughness.width  = 1;
    spec_roughness.height = 1;
    spec_roughness.format = TextureFormat::RINT8;
    _roughness_texture    = atcg::Texture2D::create(&roughness, spec_roughness);
}

void Material::setMetallic(const float metallic)
{
    TextureSpecification spec_metallic;
    spec_metallic.width  = 1;
    spec_metallic.height = 1;
    spec_metallic.format = TextureFormat::RINT8;
    _metallic_texture    = atcg::Texture2D::create(&metallic, spec_metallic);
}
}    // namespace atcg