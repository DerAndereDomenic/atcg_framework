#pragma once

#include <Material/MicrofacetMaterial.h>
#include <Material/MaterialRegistry.h>

namespace atcg
{
class ATCG_API OpaqueMaterial : public MicrofacetMaterial
{
public:
    OpaqueMaterial();

    /**
     * @brief Get the normal texture.
     *
     * @return The normal texture
     */
    ATCG_INLINE atcg::ref_ptr<atcg::Texture2D> getNormalTexture() const { return _normal_texture; }

    /**
     * @brief Get the metallic texture.
     *
     * @return The metallic texture
     */
    ATCG_INLINE atcg::ref_ptr<atcg::Texture2D> getMetallicTexture() const { return _metallic_texture; }

    /**
     * @brief Set the normal texture.
     *
     * @param texture The normal texture
     */
    ATCG_INLINE void setNormalTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _normal_texture = texture; }

    /**
     * @brief Set the metallic texture.
     *
     * @param texture The metallic texture
     */
    ATCG_INLINE void setMetallicTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _metallic_texture = texture; }

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

    /**
     * @brief Upload the material to a shader
     *
     * @param renderer The renderer
     * @param shader The shader
     */
    virtual void uploadMaterial(RendererSystem* renderer, const atcg::ref_ptr<Shader>& shader) override;

    virtual atcg::ref_ptr<Material> clone() const override;

    static void registerMaterial(MaterialRegistry::Registry* registry);

private:
    atcg::ref_ptr<atcg::Texture2D> _normal_texture;
    atcg::ref_ptr<atcg::Texture2D> _metallic_texture;
};

template<>
struct ATCG_API MaterialSerializer<OpaqueMaterial>
{
    static void serialize(const atcg::ref_ptr<OpaqueMaterial>& material, const std::filesystem::path& path);

    static atcg::ref_ptr<OpaqueMaterial> deserialize(const std::filesystem::path& path,
                                                     const nlohmann::json& material_node);
};

template<>
struct ATCG_API MaterialGUIRenderer<OpaqueMaterial>
{
    static bool renderGUI(const atcg::ref_ptr<OpaqueMaterial>& material, const std::string& key, bool& deactivated);
};


}    // namespace atcg