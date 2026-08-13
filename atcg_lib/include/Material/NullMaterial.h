#pragma once

#include <Material/Material.h>
#include <Material/MaterialRegistry.h>

namespace atcg
{
class ATCG_API NullMaterial : public Material
{
public:
    NullMaterial(const atcg::Dictionary& dict);

    ~NullMaterial();

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
};

template<>
struct ATCG_API MaterialSerializer<NullMaterial>
{
    static void serialize(const atcg::ref_ptr<NullMaterial>& material, const std::filesystem::path& path);

    static atcg::ref_ptr<NullMaterial> deserialize(const std::filesystem::path& path,
                                                   const nlohmann::json& material_node);
};

template<>
struct ATCG_API MaterialGUIRenderer<NullMaterial>
{
    static bool renderGUI(const atcg::ref_ptr<NullMaterial>& material, const std::string& key, bool& deactivated);
};

}    // namespace atcg