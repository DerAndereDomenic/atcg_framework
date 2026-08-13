#pragma once

#include <Material/MicrofacetMaterial.h>
#include <Material/MaterialRegistry.h>

namespace atcg
{
class ATCG_API DielectricMaterial : public MicrofacetMaterial
{
public:
    DielectricMaterial(const atcg::Dictionary& dict);

    virtual ~DielectricMaterial();

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
struct ATCG_API MaterialSerializer<DielectricMaterial>
{
    static void serialize(const atcg::ref_ptr<DielectricMaterial>& material, const std::filesystem::path& path);

    static atcg::ref_ptr<DielectricMaterial> deserialize(const std::filesystem::path& path,
                                                         const nlohmann::json& material_node);
};

template<>
struct ATCG_API MaterialGUIRenderer<DielectricMaterial>
{
    static bool renderGUI(const atcg::ref_ptr<DielectricMaterial>& material, const std::string& key, bool& deactivated);
};


}