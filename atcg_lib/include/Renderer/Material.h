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

    inline void setDiffuseTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _diffuse_texture = texture; }

    inline void setNormalTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _normal_texture = texture; }

    inline void setRoughnessTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _roughness_texture = texture; }

    inline void setMetallicTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _metallic_texture = texture; }

    void setDiffuseColor(const glm::vec4& color);

    void setRoughness(const float roughness);

    void setMetallic(const float metallic);

    void removeNormalMap();

private:
    atcg::ref_ptr<atcg::Texture2D> _diffuse_texture;
    atcg::ref_ptr<atcg::Texture2D> _normal_texture;
    atcg::ref_ptr<atcg::Texture2D> _roughness_texture;
    atcg::ref_ptr<atcg::Texture2D> _metallic_texture;
};
}