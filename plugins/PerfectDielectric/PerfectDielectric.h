#pragma once

#include <Renderer/Material.h>
#include <Plugin/Plugin.h>

namespace atcg
{
class PerfectDielectric : public atcg::Material
{
public:
    PerfectDielectric();

    virtual void uploadMaterial(atcg::RendererSystem* renderer, const atcg::ref_ptr<atcg::Shader>& shader) override;

    virtual atcg::ref_ptr<atcg::Material> clone() const override;

    void setIOR(const float& ior);

    void setIORTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _ior_texture = texture; }

    atcg::ref_ptr<atcg::Texture2D> getIORTexture() const { return _ior_texture; }

private:
    atcg::ref_ptr<atcg::Texture2D> _ior_texture;
};

template<>
struct MaterialGUIRenderer<PerfectDielectric>
{
    static bool renderGUI(const atcg::ref_ptr<PerfectDielectric>& material, const std::string& key, bool& deactivated);
};

template<>
struct MaterialSerializer<PerfectDielectric>
{
    static void serialize(const atcg::ref_ptr<PerfectDielectric>& material, const std::filesystem::path& path);

    static atcg::ref_ptr<PerfectDielectric> deserialize(const std::filesystem::path& path,
                                                        const nlohmann::json& material_node);
};
}    // namespace atcg