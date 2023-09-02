#include <Scene/Components.h>

namespace atcg
{
void TransformComponent::calculateModelMatrix()
{
    glm::mat4 scale     = glm::scale(_scale);
    glm::mat4 translate = glm::translate(_position);
    glm::mat4 rotation  = glm::eulerAngleXYZ(_rotation.x, _rotation.y, _rotation.z);

    _model_matrix = translate * rotation * scale;
}

void TransformComponent::decomposeModelMatrix()
{
    _position     = _model_matrix[3];
    glm::mat4 RS  = glm::mat3(_model_matrix);
    float scale_x = glm::length(RS[0]);
    float scale_y = glm::length(RS[1]);
    float scale_z = glm::length(RS[2]);
    _scale        = glm::vec3(scale_x, scale_y, scale_z);
    glm::extractEulerAngleXYZ(glm::mat4(RS * glm::scale(1.0f / _scale)), _rotation.x, _rotation.y, _rotation.z);
}

MaterialComponent::MaterialComponent()
{
    TextureSpecification spec_diffuse;
    spec_diffuse.width  = 1;
    spec_diffuse.height = 1;
    glm::u8vec3 white(255);
    _diffuse_texture = atcg::Texture2D::create(&white, spec_diffuse);

    TextureSpecification spec_normal;
    spec_normal.width  = 1;
    spec_normal.height = 1;
    glm::u8vec3 normal(0, 0, 255);
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

    TextureSpecification spec_displacement;
    spec_displacement.width  = 1;
    spec_displacement.height = 1;
    spec_displacement.format = TextureFormat::RINT8;
    uint8_t displacement     = 0;
    _displacement_texture    = atcg::Texture2D::create(&displacement, spec_displacement);
}

void MaterialComponent::setDiffuseColor(const glm::vec3& color)
{
    TextureSpecification spec_diffuse;
    spec_diffuse.width  = 1;
    spec_diffuse.height = 1;
    _diffuse_texture    = atcg::Texture2D::create(&color, spec_diffuse);
}

void MaterialComponent::setRoughness(const float roughness)
{
    TextureSpecification spec_roughness;
    spec_roughness.width  = 1;
    spec_roughness.height = 1;
    spec_roughness.format = TextureFormat::RINT8;
    _roughness_texture    = atcg::Texture2D::create(&roughness, spec_roughness);
}

void MaterialComponent::setMetallic(const float metallic)
{
    TextureSpecification spec_metallic;
    spec_metallic.width  = 1;
    spec_metallic.height = 1;
    spec_metallic.format = TextureFormat::RINT8;
    _metallic_texture    = atcg::Texture2D::create(&metallic, spec_metallic);
}

}    // namespace atcg