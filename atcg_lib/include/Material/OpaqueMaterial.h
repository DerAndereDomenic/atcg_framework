#pragma once

#include <Material/MicrofacetMaterial.h>
#include <Material/MaterialRegistry.h>

namespace atcg
{
class ATCG_API OpaqueMaterial : public MicrofacetMaterial
{
public:
    OpaqueMaterial(const atcg::Dictionary& dict);

    ~OpaqueMaterial();

    /**
     * @brief Get the normal texture.
     *
     * @return The normal texture
     */
    atcg::ref_ptr<atcg::Texture2D> getNormalTexture() const;

    /**
     * @brief Get the metallic texture.
     *
     * @return The metallic texture
     */
    atcg::ref_ptr<atcg::Texture2D> getMetallicTexture() const;
    /**
     * @brief Set the normal texture.
     *
     * @param texture The normal texture
     */
    void setNormalTexture(const atcg::ref_ptr<atcg::Texture2D>& texture);

    /**
     * @brief Set the metallic texture.
     *
     * @param texture The metallic texture
     */
    void setMetallicTexture(const atcg::ref_ptr<atcg::Texture2D>& texture);

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

    /**
     * @brief Update the data and upload it to the GPU (if necessary)
     */
    virtual void updateData() override;

    /**
     * @brief Initialize the component in the raytracing pipeline
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

    virtual atcg::ref_ptr<Material> clone() const override;

    static void registerMaterial(MaterialRegistry::Registry* registry);

private:
    class Impl;
    std::unique_ptr<Impl> impl;
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