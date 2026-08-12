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

    atcg::ref_ptr<Texture3D> density() const;

    atcg::ref_ptr<Texture3D> albedo() const;

    atcg::ref_ptr<Texture3D> emission() const;

    virtual void
    uploadMedium(RendererSystem* renderer, const atcg::ref_ptr<Shader>& shader, const glm::mat4& model) override;

    virtual atcg::ref_ptr<Medium> clone() const override;

    static void registerMedium(MediumRegistry::Registry* registry);

    struct GridComponent
    {
        BoundingBox bbox;
        AssetHandle handle = 0;
        float scale        = 1.0f;
    };

    GridComponent density_grid;
    GridComponent albedo_grid;
    GridComponent emission_grid;

private:
    atcg::ref_ptr<Texture3D> _default_emission_texture;
    atcg::ref_ptr<Texture3D> _default_albedo_texture;
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