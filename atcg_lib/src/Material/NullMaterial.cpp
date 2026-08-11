#include <Material/NullMaterial.h>
#include <Renderer/GraphicsAPI.h>
#include <Renderer/Renderer.h>
#include <Core/Application.h>

#ifndef ATCG_HEADLESS
    #include <imgui.h>
#endif

#include <portable-file-dialogs.h>

#define DIFFUSE_KEY           "Diffuse"
#define DIFFUSE_TEXTURE_KEY   "DiffuseTexture"
#define ROUGHNESS_KEY         "Roughness"
#define ROUGHNESS_TEXTURE_KEY "RoughnessTexture"
#define IOR_KEY               "IoR"
#define IOR_TEXTURE_KEY       "IoRTexture"
#define TYPE_KEY              "Type"

namespace atcg
{
NullMaterial::NullMaterial() : Material("Null")
{
    _flags = MaterialFlag::IdealTransmission | MaterialFlag::NullTransmission;
}

void NullMaterial::uploadMaterial(RendererSystem* renderer, const atcg::ref_ptr<Shader>& shader)
{
    ATCG_ASSERT(!_uploaded, "Material was already uploaded");

    uint32_t diffuse_id = renderer->popTextureID();
    // GraphicsCommand::bindTexture(diffuse_id, getDiffuseTexture());
    // shader->setInt("texture_diffuse", diffuse_id);
    _used_texture_ids[0] = diffuse_id;

    uint32_t normal_id = renderer->popTextureID();
    // GraphicsCommand::bindTexture(normal_id, getNormalTexture());
    // shader->setInt("texture_normal", normal_id);
    _used_texture_ids[1] = normal_id;

    uint32_t roughness_id = renderer->popTextureID();
    // GraphicsCommand::bindTexture(roughness_id, getRoughnessTexture());
    // shader->setInt("texture_roughness", roughness_id);
    _used_texture_ids[2] = roughness_id;

    uint32_t metallic_id = renderer->popTextureID();
    // GraphicsCommand::bindTexture(metallic_id, getMetallicTexture());
    // shader->setInt("texture_metallic", metallic_id);
    _used_texture_ids[3] = metallic_id;

    uint32_t ior_id = renderer->popTextureID();
    // GraphicsCommand::bindTexture(ior_id, getIorTexture());
    // shader->setInt("texture_ior", ior_id);
    _used_texture_ids[4] = ior_id;


    shader->selectSubroutine("sr_eval_brdf", "eval_brdf_null");
    shader->selectSubroutine("sr_image_based_lighting", "image_based_lighting_null");


    _uploaded = true;
}

atcg::ref_ptr<Material> NullMaterial::clone() const
{
    atcg::ref_ptr<NullMaterial> material = atcg::make_ref<NullMaterial>();

    return material;
}

void NullMaterial::registerMaterial(MaterialRegistry::Registry* registry)
{
    ATCG_REGISTER_MATERIAL(registry, "Null", NullMaterial);
}

bool MaterialGUIRenderer<NullMaterial>::renderGUI(const atcg::ref_ptr<NullMaterial>& material,
                                                  const std::string& key,
                                                  bool& deactivated)
{
    return false;
}

void MaterialSerializer<NullMaterial>::serialize(const atcg::ref_ptr<NullMaterial>& material,
                                                 const std::filesystem::path& path)
{
    nlohmann::json material_json;

    material_json["Version"] = "1.0";

    material_json[TYPE_KEY] = material->getMaterialType();

    std::ofstream o(path);
    o << std::setw(4) << material_json << std::endl;
}

atcg::ref_ptr<NullMaterial> MaterialSerializer<NullMaterial>::deserialize(const std::filesystem::path& path,
                                                                          const nlohmann::json& material_node)
{
    return atcg::make_ref<NullMaterial>();
}
}    // namespace atcg