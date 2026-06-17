#pragma once

#include <Renderer/Material.h>
#include <Plugin/Plugin.h>

namespace atcg
{
class DiffuseMaterial : public atcg::Material
{
public:
    ATCG_PLUGIN_CLASS(DiffuseMaterial,
                      "1.0.0",
                      "Domenic Zingsheim",
                      "A simple test material modeling a perfect diffuse reflector.");

    static atcg::ref_ptr<Material> create();

    DiffuseMaterial();

    virtual void uploadMaterial(atcg::RendererSystem* renderer, const atcg::ref_ptr<atcg::Shader>& shader) override;

    virtual atcg::ref_ptr<atcg::Material> clone() const override;

    void setDiffuseColor(const glm::vec3& color);

    void setDiffuseTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _diffuse_texture = texture; }

    atcg::ref_ptr<atcg::Texture2D> getDiffuseTexture() const { return _diffuse_texture; }

private:
    atcg::ref_ptr<atcg::Texture2D> _diffuse_texture;
};

template<>
struct atcg::MaterialGUIRenderer<DiffuseMaterial>
{
    static bool renderGUI(const atcg::ref_ptr<DiffuseMaterial>& material, const std::string& key);
};

template<>
struct MaterialSerializer<DiffuseMaterial>
{
    static void serialize(const atcg::ref_ptr<DiffuseMaterial>& material, const std::filesystem::path& path) {}

    static atcg::ref_ptr<DiffuseMaterial> deserialize(const std::filesystem::path& path,
                                                      const nlohmann::json& material_node)
    {
        return nullptr;
    }
};
}    // namespace atcg