#pragma once

#include <Renderer/Texture.h>
#include <Asset/Asset.h>

namespace atcg
{
/**
 * @brief A class to model a material.
 */
struct Material : public Asset
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
    ATCG_INLINE atcg::ref_ptr<atcg::Texture2D> getDiffuseTexture() const { return _diffuse_texture; }

    /**
     * @brief Get the normal texture.
     *
     * @return The normal texture
     */
    ATCG_INLINE atcg::ref_ptr<atcg::Texture2D> getNormalTexture() const { return _normal_texture; }

    /**
     * @brief Get the roughness texture.
     *
     * @return The roughness texture
     */
    ATCG_INLINE atcg::ref_ptr<atcg::Texture2D> getRoughnessTexture() const { return _roughness_texture; }

    /**
     * @brief Get the metallic texture.
     *
     * @return The metallic texture
     */
    ATCG_INLINE atcg::ref_ptr<atcg::Texture2D> getMetallicTexture() const { return _metallic_texture; }

    /**
     * @brief Set the diffuse texture.
     *
     * @param texture The diffuse texture
     */
    ATCG_INLINE void setDiffuseTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _diffuse_texture = texture; }

    /**
     * @brief Set the normal texture.
     *
     * @param texture The normal texture
     */
    ATCG_INLINE void setNormalTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _normal_texture = texture; }

    /**
     * @brief Set the roughness texture.
     *
     * @param texture The roughness texture
     */
    ATCG_INLINE void setRoughnessTexture(const atcg::ref_ptr<atcg::Texture2D>& texture)
    {
        _roughness_texture = texture;
    }

    /**
     * @brief Set the metallic texture.
     *
     * @param texture The metallic texture
     */
    ATCG_INLINE void setMetallicTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _metallic_texture = texture; }

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

    ATCG_INLINE static AssetType getStaticType() { return AssetType::Material; }

    ATCG_INLINE virtual AssetType getType() const override { return getStaticType(); }

private:
    atcg::ref_ptr<atcg::Texture2D> _diffuse_texture;
    atcg::ref_ptr<atcg::Texture2D> _normal_texture;
    atcg::ref_ptr<atcg::Texture2D> _roughness_texture;
    atcg::ref_ptr<atcg::Texture2D> _metallic_texture;
};
}    // namespace atcg