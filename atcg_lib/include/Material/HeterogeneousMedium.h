#pragma once

#include <Material/Medium.h>
#include <Material/MediumRegistry.h>
#include <Renderer/Texture.h>
#include <DataStructure/BoundingBox.h>

namespace atcg
{
class HeterogeneousMedium : public Medium
{
public:
    HeterogeneousMedium(const Dictionary& dict);

    ~HeterogeneousMedium();

    atcg::ref_ptr<Texture3D> density() const;

    atcg::ref_ptr<Texture3D> albedo() const;

    atcg::ref_ptr<Texture3D> emission() const;

    virtual void
    uploadMedium(RendererSystem* renderer, const atcg::ref_ptr<Shader>& shader, const glm::mat4& model) override;

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

    virtual atcg::ref_ptr<Medium> clone() const override;

    static void registerMedium(MediumRegistry::Registry* registry);

    struct GridComponent
    {
        BoundingBox bbox;
        AssetHandle handle = 0;
        float scale        = 1.0f;
    };

    GridComponent& densityGrid() const;
    GridComponent& albedoGrid() const;
    GridComponent& emissionGrid() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

template<>
struct ATCG_API MediumSerializer<HeterogeneousMedium>
{
    static void serialize(const atcg::ref_ptr<HeterogeneousMedium>& medium, const std::filesystem::path& path);

    static atcg::ref_ptr<HeterogeneousMedium> deserialize(const std::filesystem::path& path,
                                                          const nlohmann::json& medium_node);
};

template<>
struct ATCG_API MediumGUIRenderer<HeterogeneousMedium>
{
    static bool renderGUI(const atcg::ref_ptr<HeterogeneousMedium>& medium, const std::string& key, bool& deactivated);
};

}    // namespace atcg