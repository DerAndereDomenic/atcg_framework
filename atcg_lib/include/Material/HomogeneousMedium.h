#pragma once

#include <Material/Medium.h>
#include <Material/MediumRegistry.h>

namespace atcg
{
class HomogeneousMedium : public Medium
{
public:
    HomogeneousMedium(const Dictionary& dict);

    ~HomogeneousMedium();

    void setAlbedo(const glm::vec3& albedo);

    void setDensity(const float density);

    void setLe(const float Le);

    void setLeColor(const glm::vec3& Le_color);

    glm::vec3 albedo() const;

    float density() const;

    float Le() const;

    glm::vec3 Le_color() const;

    virtual void
    uploadMedium(RendererSystem* renderer, const atcg::ref_ptr<Shader>& shader, const glm::mat4& model) override;

    virtual atcg::ref_ptr<Medium> clone() const override;

    static void registerMedium(MediumRegistry::Registry* registry);

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


private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

template<>
struct ATCG_API MediumSerializer<HomogeneousMedium>
{
    static void serialize(const atcg::ref_ptr<HomogeneousMedium>& medium, const std::filesystem::path& path);

    static atcg::ref_ptr<HomogeneousMedium> deserialize(const std::filesystem::path& path,
                                                        const nlohmann::json& medium_node);
};

template<>
struct ATCG_API MediumGUIRenderer<HomogeneousMedium>
{
    static bool renderGUI(const atcg::ref_ptr<HomogeneousMedium>& medium, const std::string& key, bool& deactivated);
};

}    // namespace atcg