#pragma once

#include <Renderer/Texture.h>

namespace atcg
{
/**
 * @brief A class to model a material.
 */
struct Material
{
    /**
     * @brief Constructor
     */
    Material();

    /**
     * @brief Get the diffuse texture.
     *
     * @return The diffuse texture
     */
    inline atcg::ref_ptr<atcg::Texture2D> getDiffuseTexture() const { return _diffuse_texture; }

    /**
     * @brief Get the normal texture.
     *
     * @return The normal texture
     */
    inline atcg::ref_ptr<atcg::Texture2D> getNormalTexture() const { return _normal_texture; }

    /**
     * @brief Get the roughness texture.
     *
     * @return The roughness texture
     */
    inline atcg::ref_ptr<atcg::Texture2D> getRoughnessTexture() const { return _roughness_texture; }

    /**
     * @brief Get the metallic texture.
     *
     * @return The metallic texture
     */
    inline atcg::ref_ptr<atcg::Texture2D> getMetallicTexture() const { return _metallic_texture; }

    /**
     * @brief Set the diffuse texture.
     *
     * @param texture The diffuse texture
     */
    inline void setDiffuseTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _diffuse_texture = texture; }

    /**
     * @brief Set the normal texture.
     *
     * @param texture The normal texture
     */
    inline void setNormalTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _normal_texture = texture; }

    /**
     * @brief Set the roughness texture.
     *
     * @param texture The roughness texture
     */
    inline void setRoughnessTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _roughness_texture = texture; }

    /**
     * @brief Set the metallic texture.
     *
     * @param texture The metallic texture
     */
    inline void setMetallicTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _metallic_texture = texture; }

    /**
     * @brief Set the diffuse color.
     *
     * @param color The color
     */
    void setDiffuseColor(const glm::vec4& color);

    /**
     * @brief Set the diffuse color.
     *
     * @param color The color
     */
    void setDiffuseColor(const glm::vec3& color);

    /**
     * @brief Set the roughness value.
     *
     * @param roughness The roughness
     */
    void setRoughness(const float roughness);

    /**
     * @brief The the metallic value.
     *
     * @param metallic The metallic value
     */
    void setMetallic(const float metallic);

    /**
     * @brief Remove the normal map
     */
    void removeNormalMap();

private:
    atcg::ref_ptr<atcg::Texture2D> _diffuse_texture;
    atcg::ref_ptr<atcg::Texture2D> _normal_texture;
    atcg::ref_ptr<atcg::Texture2D> _roughness_texture;
    atcg::ref_ptr<atcg::Texture2D> _metallic_texture;
};
}    // namespace atcg