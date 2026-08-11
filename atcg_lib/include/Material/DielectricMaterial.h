#pragma once

#include <Material/MicrofacetMaterial.h>
#include <Material/MaterialRegistry.h>

namespace atcg
{
class ATCG_API DielectricMaterial : public MicrofacetMaterial
{
public:
    DielectricMaterial();

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