#pragma once

#include <Renderer/Texture.h>

namespace atcg
{
struct Material
{
    Material();

    inline atcg::ref_ptr<atcg::Texture2D> getDiffuseTexture() const { return _diffuse_texture; }
    inline atcg::ref_ptr<atcg::Texture2D> getNormalTexture() const { return _normal_texture; }
    inline atcg::ref_ptr<atcg::Texture2D> getRoughnessTexture() const { return _roughness_texture; }
    inline atcg::ref_ptr<atcg::Texture2D> getMetallicTexture() const { return _metallic_texture; }
    inline atcg::ref_ptr<atcg::Texture2D> getEmissiveTexture() const { return _emissive_texture; }

    inline void setDiffuseTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _diffuse_texture = texture; }

    inline void setNormalTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _normal_texture = texture; }

    inline void setRoughnessTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _roughness_texture = texture; }

    inline void setMetallicTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _metallic_texture = texture; }

    inline void setEmissiveTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _emissive_texture = texture; }

    void setDiffuseColor(const glm::vec4& color);

    void setDiffuseColor(const glm::vec3& color);

    void setEmissiveColor(const glm::vec4& color);

    void setEmissiveColor(const glm::vec3& color);

    void setRoughness(const float roughness);

    void setMetallic(const float metallic);

    void removeNormalMap();

    bool glass = false;

    float ior = 1.5f;

    bool emissive = false;

    float emission_scale = 1.0f;

private:
    atcg::ref_ptr<atcg::Texture2D> _diffuse_texture;
    atcg::ref_ptr<atcg::Texture2D> _normal_texture;
    atcg::ref_ptr<atcg::Texture2D> _roughness_texture;
    atcg::ref_ptr<atcg::Texture2D> _metallic_texture;
    atcg::ref_ptr<atcg::Texture2D> _emissive_texture;
};
}    // namespace atcg