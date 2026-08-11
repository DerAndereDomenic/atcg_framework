#include <Material/MicrofacetMaterial.h>

namespace atcg
{
MicrofacetMaterial::MicrofacetMaterial(const std::string& type, const atcg::Dictionary& dict) : Material(type, dict)
{
    TextureSpecification spec_diffuse;
    spec_diffuse.width  = 1;
    spec_diffuse.height = 1;
    glm::u8vec4 white(255);
    _diffuse_texture = atcg::Texture2D::create(&white, spec_diffuse);

    TextureSpecification spec_roughness;
    spec_roughness.width  = 1;
    spec_roughness.height = 1;
    spec_roughness.format = TextureFormat::RFLOAT;
    float roughness       = 1.0f;
    _roughness_texture    = atcg::Texture2D::create(&roughness, spec_roughness);

    TextureSpecification spec_ior;
    spec_ior.width  = 1;
    spec_ior.height = 1;
    spec_ior.format = TextureFormat::RFLOAT;
    float ior_value = 1.45f;
    _ior_texture    = atcg::Texture2D::create(&ior_value, spec_ior);

    _diffuse_texture   = dict.getValueOr<atcg::ref_ptr<atcg::Texture2D>>("diffuse_texture", _diffuse_texture);
    _roughness_texture = dict.getValueOr<atcg::ref_ptr<atcg::Texture2D>>("roughness_texture", _roughness_texture);
    _ior_texture       = dict.getValueOr<atcg::ref_ptr<atcg::Texture2D>>("ior_texture", _ior_texture);

    _flags = MaterialFlag::GlossyReflection | MaterialFlag::DiffuseReflection;
}

void MicrofacetMaterial::setDiffuseColor(const glm::vec4& color)
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

void MicrofacetMaterial::setDiffuseColor(const glm::vec3& color)
{
    setDiffuseColor(glm::vec4(color, 1.0f));
}

void MicrofacetMaterial::setRoughness(const float roughness)
{
    TextureSpecification spec_roughness;
    spec_roughness.width  = 1;
    spec_roughness.height = 1;
    spec_roughness.format = TextureFormat::RFLOAT;
    _roughness_texture    = atcg::Texture2D::create(&roughness, spec_roughness);
}

void MicrofacetMaterial::setIor(const float ior_value)
{
    TextureSpecification spec_ior;
    spec_ior.width  = 1;
    spec_ior.height = 1;
    spec_ior.format = TextureFormat::RFLOAT;
    _ior_texture    = atcg::Texture2D::create(&ior_value, spec_ior);
}
}    // namespace atcg