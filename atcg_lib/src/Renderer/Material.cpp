#include <Core/Assert.h>
#include <Core/Application.h>
#include <Renderer/Material.h>
#include <Renderer/Renderer.h>
#include <Renderer/Shader.h>

#ifndef ATCG_HEADLESS
    #include <imgui.h>
#endif

#include <portable-file-dialogs.h>

#define DIFFUSE_KEY           "Diffuse"
#define DIFFUSE_TEXTURE_KEY   "DiffuseTexture"
#define NORMAL_TEXTURE_KEY    "NormalTexture"
#define ROUGHNESS_KEY         "Roughness"
#define ROUGHNESS_TEXTURE_KEY "RoughnessTexture"
#define METALLIC_KEY          "Metallic"
#define METALLIC_TEXTURE_KEY  "MetallicTexture"
#define IOR_KEY               "IoR"
#define IOR_TEXTURE_KEY       "IoRTexture"
#define TYPE_KEY              "Type"
#define VERTICES_KEY          "Vertices"
#define FACES_KEY             "Faces"
#define EDGES_KEY             "Edges"
#define GEOMETRY_KEY          "Geometry"

namespace atcg
{

#pragma region MaterialRegistry

namespace MaterialRegistry
{
void registerMaterial(Registry* registry, std::string_view type, MaterialFunctions functions)
{
    registry->registerType(type, std::move(functions));
}

atcg::ref_ptr<Material> createMaterial(Registry* registry, const std::string& type, const Dictionary& dict)
{
    const MaterialFunctions* desc = registry->find(type);
    return desc->builder(dict);
}

bool renderMaterialGUI(Registry* registry,
                       const atcg::ref_ptr<Material>& material,
                       const std::string& key,
                       bool& deactivated)
{
    const MaterialFunctions* desc = registry->find(material->getMaterialType());
    return desc->gui_function(material, key, deactivated);
}

const std::vector<std::string>& getRegisteredMaterialTypes(Registry* registry)
{
    return registry->getRegisteredTypes();
}

void serializeMaterial(Registry* registry, const atcg::ref_ptr<Material>& material, const std::filesystem::path& path)
{
    const MaterialFunctions* desc = registry->find(material->getMaterialType());
    desc->serializer_function(material, path);
}

atcg::ref_ptr<Material> deserializeMaterial(Registry* registry,
                                            std::string_view material_type,
                                            const std::filesystem::path& path,
                                            const nlohmann::json& material_node)
{
    const MaterialFunctions* desc = registry->find(material_type);
    return desc->deserializer_function(path, material_node);
}
}    // namespace MaterialRegistry

#pragma endregion

#pragma region Materials

Material::Material(const std::string& type) : _material_type(type) {}

void Material::releaseTextureIDs(RendererSystem* renderer)
{
    ATCG_ASSERT(_uploaded, "Tried freeing material ids without while material is not uploaded");

    renderer->pushTextureID(_used_texture_ids[0]);
    renderer->pushTextureID(_used_texture_ids[1]);
    renderer->pushTextureID(_used_texture_ids[2]);
    renderer->pushTextureID(_used_texture_ids[3]);
    renderer->pushTextureID(_used_texture_ids[4]);

    _uploaded = false;
}

MicrofacetMaterial::MicrofacetMaterial(const std::string& type) : Material(type)
{
    TextureSpecification spec_diffuse;
    spec_diffuse.width  = 1;
    spec_diffuse.height = 1;
    glm::u8vec4 white(255);
    _diffuse_texture = atcg::Texture2D::create(&white, spec_diffuse);

    TextureSpecification spec_roughness;
    spec_roughness.width  = 1;
    spec_roughness.height = 1;
    spec_roughness.format = TextureFormat::RFLOAT;
    float roughness       = 1.0f;
    _roughness_texture    = atcg::Texture2D::create(&roughness, spec_roughness);

    TextureSpecification spec_ior;
    spec_ior.width  = 1;
    spec_ior.height = 1;
    spec_ior.format = TextureFormat::RFLOAT;
    float ior_value = 1.45f;
    _ior_texture    = atcg::Texture2D::create(&ior_value, spec_ior);
}

void MicrofacetMaterial::setDiffuseColor(const glm::vec4& color)
{
    TextureSpecification spec_diffuse;
    spec_diffuse.width  = 1;
    spec_diffuse.height = 1;
    glm::u8vec4 color_quant((uint8_t)(color[0] * 255.0f),
                            (uint8_t)(color[1] * 255.0f),
                            (uint8_t)(color[2] * 255.0f),
                            (uint8_t)(color[3] * 255.0f));
    _diffuse_texture = atcg::Texture2D::create(&color_quant, spec_diffuse);
}

void MicrofacetMaterial::setDiffuseColor(const glm::vec3& color)
{
    setDiffuseColor(glm::vec4(color, 1.0f));
}

void MicrofacetMaterial::setRoughness(const float roughness)
{
    TextureSpecification spec_roughness;
    spec_roughness.width  = 1;
    spec_roughness.height = 1;
    spec_roughness.format = TextureFormat::RFLOAT;
    _roughness_texture    = atcg::Texture2D::create(&roughness, spec_roughness);
}

void MicrofacetMaterial::setIor(const float ior_value)
{
    TextureSpecification spec_ior;
    spec_ior.width  = 1;
    spec_ior.height = 1;
    spec_ior.format = TextureFormat::RFLOAT;
    _ior_texture    = atcg::Texture2D::create(&ior_value, spec_ior);
}

OpaqueMaterial::OpaqueMaterial() : MicrofacetMaterial("Opaque")
{
    TextureSpecification spec_normal;
    spec_normal.width  = 1;
    spec_normal.height = 1;
    glm::u8vec4 normal(127, 127, 255, 255);
    _normal_texture = atcg::Texture2D::create(&normal, spec_normal);

    TextureSpecification spec_metallic;
    spec_metallic.width  = 1;
    spec_metallic.height = 1;
    spec_metallic.format = TextureFormat::RFLOAT;
    float metallic       = 0.0f;
    _metallic_texture    = atcg::Texture2D::create(&metallic, spec_metallic);
}

void OpaqueMaterial::setMetallic(const float metallic)
{
    TextureSpecification spec_metallic;
    spec_metallic.width  = 1;
    spec_metallic.height = 1;
    spec_metallic.format = TextureFormat::RFLOAT;
    _metallic_texture    = atcg::Texture2D::create(&metallic, spec_metallic);
}

void OpaqueMaterial::removeNormalMap()
{
    TextureSpecification spec_normal;
    spec_normal.width  = 1;
    spec_normal.height = 1;
    glm::u8vec4 normal(127, 127, 255, 255);
    _normal_texture = atcg::Texture2D::create(&normal, spec_normal);
}

void OpaqueMaterial::uploadMaterial(RendererSystem* renderer, const atcg::ref_ptr<Shader>& shader)
{
    ATCG_ASSERT(!_uploaded, "Material was already uploaded");

    uint32_t diffuse_id = renderer->popTextureID();
    GraphicsCommand::bindTexture(diffuse_id, getDiffuseTexture());
    shader->setInt("texture_diffuse", diffuse_id);
    _used_texture_ids[0] = diffuse_id;

    uint32_t normal_id = renderer->popTextureID();
    GraphicsCommand::bindTexture(normal_id, getNormalTexture());
    shader->setInt("texture_normal", normal_id);
    _used_texture_ids[1] = normal_id;

    uint32_t roughness_id = renderer->popTextureID();
    GraphicsCommand::bindTexture(roughness_id, getRoughnessTexture());
    shader->setInt("texture_roughness", roughness_id);
    _used_texture_ids[2] = roughness_id;

    uint32_t metallic_id = renderer->popTextureID();
    GraphicsCommand::bindTexture(metallic_id, getMetallicTexture());
    shader->setInt("texture_metallic", metallic_id);
    _used_texture_ids[3] = metallic_id;

    uint32_t ior_id = renderer->popTextureID();
    GraphicsCommand::bindTexture(ior_id, getIorTexture());
    shader->setInt("texture_ior", ior_id);
    _used_texture_ids[4] = ior_id;

    // Select shading functions
    shader->selectSubroutine("sr_eval_brdf", "eval_brdf_pbr");
    shader->selectSubroutine("sr_image_based_lighting", "image_based_lighting_pbr");

    _uploaded = true;
}

atcg::ref_ptr<Material> OpaqueMaterial::clone() const
{
    atcg::ref_ptr<OpaqueMaterial> material = atcg::make_ref<OpaqueMaterial>();

    material->setDiffuseTexture(std::dynamic_pointer_cast<atcg::Texture2D>(getDiffuseTexture()->clone()));
    material->setRoughnessTexture(std::dynamic_pointer_cast<atcg::Texture2D>(getRoughnessTexture()->clone()));
    material->setIorTexture(std::dynamic_pointer_cast<atcg::Texture2D>(getIorTexture()->clone()));
    material->setMetallicTexture(std::dynamic_pointer_cast<atcg::Texture2D>(getMetallicTexture()->clone()));
    material->setNormalTexture(std::dynamic_pointer_cast<atcg::Texture2D>(getNormalTexture()->clone()));

    return material;
}

void OpaqueMaterial::registerMaterial(MaterialRegistry::Registry* registry)
{
    ATCG_REGISTER_MATERIAL(registry, "Opaque", OpaqueMaterial);
}

DielectricMaterial::DielectricMaterial() : MicrofacetMaterial("Dielectric") {}

void DielectricMaterial::uploadMaterial(RendererSystem* renderer, const atcg::ref_ptr<Shader>& shader)
{
    ATCG_ASSERT(!_uploaded, "Material was already uploaded");

    uint32_t diffuse_id = renderer->popTextureID();
    GraphicsCommand::bindTexture(diffuse_id, getDiffuseTexture());
    shader->setInt("texture_diffuse", diffuse_id);
    _used_texture_ids[0] = diffuse_id;

    // TODO: Not used but the ids need to be valid for the release function
    uint32_t normal_id = renderer->popTextureID();
    // GraphicsCommand::bindTexture(normal_id, getNormalTexture());
    // shader->setInt("texture_normal", normal_id);
    _used_texture_ids[1] = normal_id;

    uint32_t roughness_id = renderer->popTextureID();
    GraphicsCommand::bindTexture(roughness_id, getRoughnessTexture());
    shader->setInt("texture_roughness", roughness_id);
    _used_texture_ids[2] = roughness_id;

    uint32_t metallic_id = renderer->popTextureID();
    // GraphicsCommand::bindTexture(metallic_id, getMetallicTexture());
    // shader->setInt("texture_metallic", metallic_id);
    _used_texture_ids[3] = metallic_id;

    uint32_t ior_id = renderer->popTextureID();
    GraphicsCommand::bindTexture(ior_id, getIorTexture());
    shader->setInt("texture_ior", ior_id);
    _used_texture_ids[4] = ior_id;

    shader->selectSubroutine("sr_eval_brdf", "eval_brdf_glass");
    shader->selectSubroutine("sr_image_based_lighting", "image_based_lighting_glass");

    _uploaded = true;
}

atcg::ref_ptr<Material> DielectricMaterial::clone() const
{
    atcg::ref_ptr<DielectricMaterial> material = atcg::make_ref<DielectricMaterial>();

    material->setDiffuseTexture(std::dynamic_pointer_cast<atcg::Texture2D>(getDiffuseTexture()->clone()));
    material->setRoughnessTexture(std::dynamic_pointer_cast<atcg::Texture2D>(getRoughnessTexture()->clone()));
    material->setIorTexture(std::dynamic_pointer_cast<atcg::Texture2D>(getIorTexture()->clone()));

    return material;
}

void DielectricMaterial::registerMaterial(MaterialRegistry::Registry* registry)
{
    ATCG_REGISTER_MATERIAL(registry, "Dielectric", DielectricMaterial);
}

NullMaterial::NullMaterial() : Material("Null") {}

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


#pragma endregion

#pragma region MaterialGUIRenderers

bool MaterialGUIRenderer<OpaqueMaterial>::renderGUI(const atcg::ref_ptr<OpaqueMaterial>& material,
                                                    const std::string& key,
                                                    bool& deactivated)
{
    bool updated = false;
#ifndef ATCG_HEADLESS
    float content_scale = atcg::Application::get()->getWindow()->getContentScale();
    {
        auto spec        = material->getDiffuseTexture()->getSpecification();
        bool useTextures = spec.width != 1 || spec.height != 1;

        if(!useTextures)
        {
            auto diffuse = material->getDiffuseTexture()->getData(atcg::CPU);

            float color[4] = {diffuse.index({0, 0, 0}).item<float>() / 255.0f,
                              diffuse.index({0, 0, 1}).item<float>() / 255.0f,
                              diffuse.index({0, 0, 2}).item<float>() / 255.0f,
                              diffuse.index({0, 0, 3}).item<float>() / 255.0f};

            if(ImGui::ColorEdit4(("Diffuse##" + key).c_str(), color))
            {
                glm::vec4 new_color = glm::make_vec4(color);
                material->setDiffuseColor(new_color);
                updated = true;
            }
            deactivated = ImGui::IsItemDeactivated() || deactivated;

            ImGui::SameLine();

            if(ImGui::Button(("...##diffuse" + key).c_str()))
            {
                auto f     = pfd::open_file("Choose files to read",
                                            pfd::path::home(),
                                            {"All Files",
                                             "*",
                                             "PNG Files (.png)",
                                             "*.png",
                                             "JPG Files (.jpg, .jpeg)",
                                             "*jpg, *jpeg",
                                             "BMP Files (.bmp)",
                                             "*.bmp",
                                             "HDR Files (.hdr)",
                                             "*.hdr"},
                                            pfd::opt::none);
                auto files = f.result();
                if(!files.empty())
                {
                    auto img     = IO::imread(files[0], 2.2f);
                    auto texture = atcg::Texture2D::create(img);
                    material->setDiffuseTexture(texture);
                    updated = true;
                }
            }
            deactivated = ImGui::IsItemDeactivated() || deactivated;
        }
        else
        {
            ImGui::Text("Diffuse Texture");
            ImGui::SameLine();

            if(ImGui::Button(("X##diffuse" + key).c_str()))
            {
                material->setDiffuseColor(glm::vec4(1));
                updated = true;
            }
            else
                ImGui::Image((ImTextureID)material->getDiffuseTexture()->getID(),
                             ImVec2(content_scale * 128, content_scale * 128),
                             ImVec2 {0, 1},
                             ImVec2 {1, 0});
            deactivated = ImGui::IsItemDeactivated() || deactivated;
        }
    }

    {
        auto spec        = material->getNormalTexture()->getSpecification();
        bool useTextures = spec.width != 1 || spec.height != 1;

        if(!useTextures)
        {
            ImGui::Text("Normals");
            ImGui::SameLine();
            if(ImGui::Button(("...##normals" + key).c_str()))
            {
                auto f     = pfd::open_file("Choose files to read",
                                            pfd::path::home(),
                                            {"All Files",
                                             "*",
                                             "PNG Files (.png)",
                                             "*.png",
                                             "JPG Files (.jpg, .jpeg)",
                                             "*jpg, *jpeg",
                                             "BMP Files (.bmp)",
                                             "*.bmp",
                                             "HDR Files (.hdr)",
                                             "*.hdr"},
                                            pfd::opt::none);
                auto files = f.result();
                if(!files.empty())
                {
                    auto img     = IO::imread(files[0]);
                    auto texture = atcg::Texture2D::create(img);
                    material->setNormalTexture(texture);
                    updated = true;
                }
            }
            deactivated = ImGui::IsItemDeactivated() || deactivated;
        }
        else
        {
            ImGui::Text("Normal Texture");
            ImGui::SameLine();

            if(ImGui::Button(("X##normal" + key).c_str()))
            {
                material->removeNormalMap();
                updated = true;
            }
            else
                ImGui::Image((ImTextureID)material->getNormalTexture()->getID(),
                             ImVec2(content_scale * 128, content_scale * 128),
                             ImVec2 {0, 1},
                             ImVec2 {1, 0});
            deactivated = ImGui::IsItemDeactivated() || deactivated;
        }
    }

    {
        auto spec        = material->getRoughnessTexture()->getSpecification();
        bool useTextures = spec.width != 1 || spec.height != 1;

        if(!useTextures)
        {
            auto data       = material->getRoughnessTexture()->getData(atcg::CPU);
            float roughness = data.item<float>();

            if(ImGui::DragFloat(("Roughness##" + key).c_str(), &roughness, 0.005f, 0.0f, 1.0f))
            {
                material->setRoughness(roughness);
                updated = true;
            }
            deactivated = ImGui::IsItemDeactivated() || deactivated;

            ImGui::SameLine();

            if(ImGui::Button(("...##roughness" + key).c_str()))
            {
                auto f     = pfd::open_file("Choose files to read",
                                            pfd::path::home(),
                                            {"All Files",
                                             "*",
                                             "PNG Files (.png)",
                                             "*.png",
                                             "JPG Files (.jpg, .jpeg)",
                                             "*jpg, *jpeg",
                                             "BMP Files (.bmp)",
                                             "*.bmp",
                                             "HDR Files (.hdr)",
                                             "*.hdr"},
                                            pfd::opt::none);
                auto files = f.result();
                if(!files.empty())
                {
                    auto img     = IO::imread(files[0]);
                    auto texture = atcg::Texture2D::create(img);
                    material->setRoughnessTexture(texture);
                    updated = true;
                }
            }
            deactivated = ImGui::IsItemDeactivated() || deactivated;
        }
        else
        {
            ImGui::Text("Roughness Texture");
            ImGui::SameLine();

            if(ImGui::Button(("X##roughness" + key).c_str()))
            {
                material->setRoughness(1.0f);
                updated = true;
            }
            else
                ImGui::Image((ImTextureID)material->getRoughnessTexture()->getID(),
                             ImVec2(content_scale * 128, content_scale * 128),
                             ImVec2 {0, 1},
                             ImVec2 {1, 0});
            deactivated = ImGui::IsItemDeactivated() || deactivated;
        }
    }

    {
        auto spec        = material->getMetallicTexture()->getSpecification();
        bool useTextures = spec.width != 1 || spec.height != 1;

        if(!useTextures)
        {
            auto data      = material->getMetallicTexture()->getData(atcg::CPU);
            float metallic = data.item<float>();

            if(ImGui::DragFloat(("Metallic##" + key).c_str(), &metallic, 0.005f, 0.0f, 1.0f))
            {
                material->setMetallic(metallic);
                updated = true;
            }
            deactivated = ImGui::IsItemDeactivated() || deactivated;

            ImGui::SameLine();

            if(ImGui::Button(("...##metallic" + key).c_str()))
            {
                auto f     = pfd::open_file("Choose files to read",
                                            pfd::path::home(),
                                            {"All Files",
                                             "*",
                                             "PNG Files (.png)",
                                             "*.png",
                                             "JPG Files (.jpg, .jpeg)",
                                             "*jpg, *jpeg",
                                             "BMP Files (.bmp)",
                                             "*.bmp",
                                             "HDR Files (.hdr)",
                                             "*.hdr"},
                                            pfd::opt::none);
                auto files = f.result();
                if(!files.empty())
                {
                    auto img     = IO::imread(files[0]);
                    auto texture = atcg::Texture2D::create(img);
                    material->setMetallicTexture(texture);
                    updated = true;
                }
            }
            deactivated = ImGui::IsItemDeactivated() || deactivated;
        }
        else
        {
            ImGui::Text("Metallic Texture");
            ImGui::SameLine();

            if(ImGui::Button(("X##metallic" + key).c_str()))
            {
                material->setMetallic(0.0f);
                updated = true;
            }
            else
                ImGui::Image((ImTextureID)material->getMetallicTexture()->getID(),
                             ImVec2(content_scale * 128, content_scale * 128),
                             ImVec2 {0, 1},
                             ImVec2 {1, 0});
            deactivated = ImGui::IsItemDeactivated() || deactivated;
        }
    }

    {
        auto spec        = material->getIorTexture()->getSpecification();
        bool useTextures = spec.width != 1 || spec.height != 1;

        if(!useTextures)
        {
            auto data = material->getIorTexture()->getData(atcg::CPU);
            float ior = data.item<float>();

            if(ImGui::DragFloat(("IoR##" + key).c_str(), &ior, 0.005f, 1.0f, 2.5f))
            {
                material->setIor(ior);
                updated = true;
            }
            deactivated = ImGui::IsItemDeactivated() || deactivated;

            ImGui::SameLine();

            if(ImGui::Button(("...##ior" + key).c_str()))
            {
                auto f     = pfd::open_file("Choose files to read",
                                            pfd::path::home(),
                                            {"All Files",
                                             "*",
                                             "PNG Files (.png)",
                                             "*.png",
                                             "JPG Files (.jpg, .jpeg)",
                                             "*jpg, *jpeg",
                                             "BMP Files (.bmp)",
                                             "*.bmp",
                                             "HDR Files (.hdr)",
                                             "*.hdr"},
                                            pfd::opt::none);
                auto files = f.result();
                if(!files.empty())
                {
                    auto img     = IO::imread(files[0]);
                    auto texture = atcg::Texture2D::create(img);
                    material->setIorTexture(texture);
                    updated = true;
                }
            }
            deactivated = ImGui::IsItemDeactivated() || deactivated;
        }
        else
        {
            ImGui::Text("IoR Texture");
            ImGui::SameLine();

            if(ImGui::Button(("X##ior" + key).c_str()))
            {
                material->setIor(1.5f);
                updated = true;
            }
            else
                ImGui::Image((ImTextureID)material->getIorTexture()->getID(),
                             ImVec2(content_scale * 128, content_scale * 128),
                             ImVec2 {0, 1},
                             ImVec2 {1, 0});
            deactivated = ImGui::IsItemDeactivated() || deactivated;
        }
    }
#endif
    return updated;
}

bool MaterialGUIRenderer<DielectricMaterial>::renderGUI(const atcg::ref_ptr<DielectricMaterial>& material,
                                                        const std::string& key,
                                                        bool& deactivated)
{
    bool updated = false;
#ifndef ATCG_HEADLESS
    float content_scale = atcg::Application::get()->getWindow()->getContentScale();
    {
        auto spec        = material->getDiffuseTexture()->getSpecification();
        bool useTextures = spec.width != 1 || spec.height != 1;

        if(!useTextures)
        {
            auto diffuse = material->getDiffuseTexture()->getData(atcg::CPU);

            float color[4] = {diffuse.index({0, 0, 0}).item<float>() / 255.0f,
                              diffuse.index({0, 0, 1}).item<float>() / 255.0f,
                              diffuse.index({0, 0, 2}).item<float>() / 255.0f,
                              diffuse.index({0, 0, 3}).item<float>() / 255.0f};

            if(ImGui::ColorEdit4(("Diffuse##" + key).c_str(), color))
            {
                glm::vec4 new_color = glm::make_vec4(color);
                material->setDiffuseColor(new_color);
                updated = true;
            }
            deactivated = ImGui::IsItemDeactivated() || deactivated;

            ImGui::SameLine();

            if(ImGui::Button(("...##diffuse" + key).c_str()))
            {
                auto f     = pfd::open_file("Choose files to read",
                                            pfd::path::home(),
                                            {"All Files",
                                             "*",
                                             "PNG Files (.png)",
                                             "*.png",
                                             "JPG Files (.jpg, .jpeg)",
                                             "*jpg, *jpeg",
                                             "BMP Files (.bmp)",
                                             "*.bmp",
                                             "HDR Files (.hdr)",
                                             "*.hdr"},
                                            pfd::opt::none);
                auto files = f.result();
                if(!files.empty())
                {
                    auto img     = IO::imread(files[0], 2.2f);
                    auto texture = atcg::Texture2D::create(img);
                    material->setDiffuseTexture(texture);
                    updated = true;
                }
            }
            deactivated = ImGui::IsItemDeactivated() || deactivated;
        }
        else
        {
            ImGui::Text("Diffuse Texture");
            ImGui::SameLine();

            if(ImGui::Button(("X##diffuse" + key).c_str()))
            {
                material->setDiffuseColor(glm::vec4(1));
                updated = true;
            }
            else
                ImGui::Image((ImTextureID)material->getDiffuseTexture()->getID(),
                             ImVec2(content_scale * 128, content_scale * 128),
                             ImVec2 {0, 1},
                             ImVec2 {1, 0});
            deactivated = ImGui::IsItemDeactivated() || deactivated;
        }
    }

    {
        auto spec        = material->getRoughnessTexture()->getSpecification();
        bool useTextures = spec.width != 1 || spec.height != 1;

        if(!useTextures)
        {
            auto data       = material->getRoughnessTexture()->getData(atcg::CPU);
            float roughness = data.item<float>();

            if(ImGui::DragFloat(("Roughness##" + key).c_str(), &roughness, 0.005f, 0.0f, 1.0f))
            {
                material->setRoughness(roughness);
                updated = true;
            }
            deactivated = ImGui::IsItemDeactivated() || deactivated;

            ImGui::SameLine();

            if(ImGui::Button(("...##roughness" + key).c_str()))
            {
                auto f     = pfd::open_file("Choose files to read",
                                            pfd::path::home(),
                                            {"All Files",
                                             "*",
                                             "PNG Files (.png)",
                                             "*.png",
                                             "JPG Files (.jpg, .jpeg)",
                                             "*jpg, *jpeg",
                                             "BMP Files (.bmp)",
                                             "*.bmp",
                                             "HDR Files (.hdr)",
                                             "*.hdr"},
                                            pfd::opt::none);
                auto files = f.result();
                if(!files.empty())
                {
                    auto img     = IO::imread(files[0]);
                    auto texture = atcg::Texture2D::create(img);
                    material->setRoughnessTexture(texture);
                    updated = true;
                }
            }
            deactivated = ImGui::IsItemDeactivated() || deactivated;
        }
        else
        {
            ImGui::Text("Roughness Texture");
            ImGui::SameLine();

            if(ImGui::Button(("X##roughness" + key).c_str()))
            {
                material->setRoughness(1.0f);
                updated = true;
            }
            else
                ImGui::Image((ImTextureID)material->getRoughnessTexture()->getID(),
                             ImVec2(content_scale * 128, content_scale * 128),
                             ImVec2 {0, 1},
                             ImVec2 {1, 0});
            deactivated = ImGui::IsItemDeactivated() || deactivated;
        }
    }

    {
        auto spec        = material->getIorTexture()->getSpecification();
        bool useTextures = spec.width != 1 || spec.height != 1;

        if(!useTextures)
        {
            auto data = material->getIorTexture()->getData(atcg::CPU);
            float ior = data.item<float>();

            if(ImGui::DragFloat(("IoR##" + key).c_str(), &ior, 0.005f, 1.0f, 2.5f))
            {
                material->setIor(ior);
                updated = true;
            }
            deactivated = ImGui::IsItemDeactivated() || deactivated;

            ImGui::SameLine();

            if(ImGui::Button(("...##ior" + key).c_str()))
            {
                auto f     = pfd::open_file("Choose files to read",
                                            pfd::path::home(),
                                            {"All Files",
                                             "*",
                                             "PNG Files (.png)",
                                             "*.png",
                                             "JPG Files (.jpg, .jpeg)",
                                             "*jpg, *jpeg",
                                             "BMP Files (.bmp)",
                                             "*.bmp",
                                             "HDR Files (.hdr)",
                                             "*.hdr"},
                                            pfd::opt::none);
                auto files = f.result();
                if(!files.empty())
                {
                    auto img     = IO::imread(files[0]);
                    auto texture = atcg::Texture2D::create(img);
                    material->setIorTexture(texture);
                    updated = true;
                }
            }
            deactivated = ImGui::IsItemDeactivated() || deactivated;
        }
        else
        {
            ImGui::Text("IoR Texture");
            ImGui::SameLine();

            if(ImGui::Button(("X##ior" + key).c_str()))
            {
                material->setIor(1.5f);
                updated = true;
            }
            else
                ImGui::Image((ImTextureID)material->getIorTexture()->getID(),
                             ImVec2(content_scale * 128, content_scale * 128),
                             ImVec2 {0, 1},
                             ImVec2 {1, 0});
            deactivated = ImGui::IsItemDeactivated() || deactivated;
        }
    }
#endif
    return updated;
}

bool MaterialGUIRenderer<NullMaterial>::renderGUI(const atcg::ref_ptr<NullMaterial>& material,
                                                  const std::string& key,
                                                  bool& deactivated)
{
    return false;
}

#pragma endregion

#pragma region MaterialSerializers

namespace detail
{
ATCG_INLINE std::string
serialize_texture2d_ver1(const atcg::ref_ptr<Texture2D>& texture, const std::filesystem::path& path, float gamma = 1.0f)
{
    torch::Tensor texture_data = texture->getData(atcg::CPU);

    Image img(texture_data);

    std::string file_ending = ".png";
    if(img.isHDR())
    {
        file_ending = ".hdr";
    }

    auto img_path = path.string() + file_ending;
    img.applyGamma(gamma);
    img.store(img_path);

    return file_ending;
}
}    // namespace detail

void MaterialSerializer<OpaqueMaterial>::serialize(const atcg::ref_ptr<OpaqueMaterial>& material,
                                                   const std::filesystem::path& path)
{
    nlohmann::json material_json;

    material_json["Version"] = "1.0";

    auto diffuse_texture   = material->getDiffuseTexture();
    auto normal_texture    = material->getNormalTexture();
    auto metallic_texture  = material->getMetallicTexture();
    auto roughness_texture = material->getRoughnessTexture();
    auto ior_texture       = material->getIorTexture();

    bool use_diffuse_texture   = !(diffuse_texture->width() == 1 && diffuse_texture->height() == 1);
    bool use_normal_texture    = !(normal_texture->width() == 1 && normal_texture->height() == 1);
    bool use_metallic_texture  = !(metallic_texture->width() == 1 && metallic_texture->height() == 1);
    bool use_roughness_texture = !(roughness_texture->width() == 1 && roughness_texture->height() == 1);
    bool use_ior_texture       = !(ior_texture->width() == 1 && ior_texture->height() == 1);

    material_json[TYPE_KEY] = material->getMaterialType();
    if(use_diffuse_texture)
    {
        std::filesystem::path img_path = path.parent_path() / "diffuse";

        auto file_ending = detail::serialize_texture2d_ver1(diffuse_texture, img_path, 1.0f / 2.2f);

        material_json[DIFFUSE_TEXTURE_KEY] = "diffuse" + file_ending;
    }
    else
    {
        auto data         = diffuse_texture->getData(atcg::CPU);
        glm::u8vec4 color = {data.index({0, 0, 0}).item<uint8_t>(),
                             data.index({0, 0, 1}).item<uint8_t>(),
                             data.index({0, 0, 2}).item<uint8_t>(),
                             data.index({0, 0, 3}).item<uint8_t>()};

        glm::vec4 c(color);
        c = c / 255.0f;

        material_json[DIFFUSE_KEY] = nlohmann::json::array({c.x, c.y, c.z, c.w});
    }

    if(use_normal_texture)
    {
        std::filesystem::path img_path = path.parent_path() / "normals";

        auto file_ending = detail::serialize_texture2d_ver1(normal_texture, img_path);

        material_json[NORMAL_TEXTURE_KEY] = "normals" + file_ending;
    }

    if(use_metallic_texture)
    {
        std::filesystem::path img_path = path.parent_path() / "metallic";

        auto file_ending = detail::serialize_texture2d_ver1(metallic_texture, img_path);

        material_json[METALLIC_TEXTURE_KEY] = "metallic" + file_ending;
    }
    else
    {
        auto data   = metallic_texture->getData(atcg::CPU);
        float color = data.item<float>();

        material_json[METALLIC_KEY] = color;
    }

    if(use_roughness_texture)
    {
        std::filesystem::path img_path = path.parent_path() / "roughness";

        auto file_ending = detail::serialize_texture2d_ver1(roughness_texture, img_path);

        material_json[ROUGHNESS_TEXTURE_KEY] = "roughness" + file_ending;
    }
    else
    {
        auto data   = roughness_texture->getData(atcg::CPU);
        float color = data.item<float>();

        material_json[ROUGHNESS_KEY] = color;
    }

    if(use_ior_texture)
    {
        std::filesystem::path img_path = path.parent_path() / "ior";

        auto file_ending = detail::serialize_texture2d_ver1(ior_texture, img_path);

        material_json[IOR_TEXTURE_KEY] = "ior" + file_ending;
    }
    else
    {
        auto data   = ior_texture->getData(atcg::CPU);
        float color = data.item<float>();

        material_json[IOR_KEY] = color;
    }

    std::ofstream o(path);
    o << std::setw(4) << material_json << std::endl;
}

atcg::ref_ptr<OpaqueMaterial> MaterialSerializer<OpaqueMaterial>::deserialize(const std::filesystem::path& path,
                                                                              const nlohmann::json& material_node)
{
    atcg::ref_ptr<OpaqueMaterial> material = atcg::make_ref<OpaqueMaterial>();

    // Diffuse
    if(material_node.contains(DIFFUSE_KEY))
    {
        std::vector<float> diffuse_color = material_node[DIFFUSE_KEY];
        if(diffuse_color.size() == 3)
        {
            material->setDiffuseColor(glm::vec4(glm::make_vec3(diffuse_color.data()), 1.0f));
        }
        else if(diffuse_color.size() == 4)
        {
            material->setDiffuseColor(glm::make_vec4(diffuse_color.data()));
        }
    }
    else if(material_node.contains(DIFFUSE_TEXTURE_KEY))
    {
        std::filesystem::path diffuse_path = path.parent_path() / material_node[DIFFUSE_TEXTURE_KEY];
        auto img                           = IO::imread(diffuse_path.generic_string(), 2.2f);
        auto diffuse_texture               = atcg::Texture2D::create(img);
        material->setDiffuseTexture(diffuse_texture);
    }

    // Normals
    if(material_node.contains(NORMAL_TEXTURE_KEY))
    {
        std::filesystem::path normal_path = path.parent_path() / material_node[NORMAL_TEXTURE_KEY];
        auto img                          = IO::imread(normal_path.generic_string());
        auto normal_texture               = atcg::Texture2D::create(img);
        material->setNormalTexture(normal_texture);
    }

    // Roughness
    if(material_node.contains(ROUGHNESS_KEY))
    {
        float roughness = material_node[ROUGHNESS_KEY];
        material->setRoughness(roughness);
    }
    else if(material_node.contains(ROUGHNESS_TEXTURE_KEY))
    {
        std::filesystem::path roughness_path = path.parent_path() / material_node[ROUGHNESS_TEXTURE_KEY];
        auto img                             = IO::imread(roughness_path.generic_string());
        auto roughness_texture               = atcg::Texture2D::create(img);
        material->setRoughnessTexture(roughness_texture);
    }

    // Metallic
    if(material_node.contains(METALLIC_KEY))
    {
        float metallic = material_node[METALLIC_KEY];
        material->setMetallic(metallic);
    }
    else if(material_node.contains(METALLIC_TEXTURE_KEY))
    {
        std::filesystem::path metallic_path = path.parent_path() / material_node[METALLIC_TEXTURE_KEY];
        auto img                            = IO::imread(metallic_path.generic_string());
        auto metallic_texture               = atcg::Texture2D::create(img);
        material->setMetallicTexture(metallic_texture);
    }

    // IoR
    if(material_node.contains(IOR_KEY))
    {
        float ior = material_node[IOR_KEY];
        material->setIor(ior);
    }
    else if(material_node.contains(IOR_TEXTURE_KEY))
    {
        std::filesystem::path ior_path = path.parent_path() / material_node[IOR_TEXTURE_KEY];
        auto img                       = IO::imread(ior_path.generic_string());
        auto ior_texture               = atcg::Texture2D::create(img);
        material->setIorTexture(ior_texture);
    }

    return material;
}

void MaterialSerializer<DielectricMaterial>::serialize(const atcg::ref_ptr<DielectricMaterial>& material,
                                                       const std::filesystem::path& path)
{
    nlohmann::json material_json;

    material_json["Version"] = "1.0";

    auto diffuse_texture   = material->getDiffuseTexture();
    auto roughness_texture = material->getRoughnessTexture();
    auto ior_texture       = material->getIorTexture();

    bool use_diffuse_texture   = !(diffuse_texture->width() == 1 && diffuse_texture->height() == 1);
    bool use_roughness_texture = !(roughness_texture->width() == 1 && roughness_texture->height() == 1);
    bool use_ior_texture       = !(ior_texture->width() == 1 && ior_texture->height() == 1);

    material_json[TYPE_KEY] = material->getMaterialType();
    if(use_diffuse_texture)
    {
        std::filesystem::path img_path = path.parent_path() / "diffuse";

        auto file_ending = detail::serialize_texture2d_ver1(diffuse_texture, img_path, 1.0f / 2.2f);

        material_json[DIFFUSE_TEXTURE_KEY] = "diffuse" + file_ending;
    }
    else
    {
        auto data         = diffuse_texture->getData(atcg::CPU);
        glm::u8vec4 color = {data.index({0, 0, 0}).item<uint8_t>(),
                             data.index({0, 0, 1}).item<uint8_t>(),
                             data.index({0, 0, 2}).item<uint8_t>(),
                             data.index({0, 0, 3}).item<uint8_t>()};

        glm::vec4 c(color);
        c = c / 255.0f;

        material_json[DIFFUSE_KEY] = nlohmann::json::array({c.x, c.y, c.z, c.w});
    }

    if(use_roughness_texture)
    {
        std::filesystem::path img_path = path.parent_path() / "roughness";

        auto file_ending = detail::serialize_texture2d_ver1(roughness_texture, img_path);

        material_json[ROUGHNESS_TEXTURE_KEY] = "roughness" + file_ending;
    }
    else
    {
        auto data   = roughness_texture->getData(atcg::CPU);
        float color = data.item<float>();

        material_json[ROUGHNESS_KEY] = color;
    }

    if(use_ior_texture)
    {
        std::filesystem::path img_path = path.parent_path() / "ior";

        auto file_ending = detail::serialize_texture2d_ver1(ior_texture, img_path);

        material_json[IOR_TEXTURE_KEY] = "ior" + file_ending;
    }
    else
    {
        auto data   = ior_texture->getData(atcg::CPU);
        float color = data.item<float>();

        material_json[IOR_KEY] = color;
    }

    std::ofstream o(path);
    o << std::setw(4) << material_json << std::endl;
}

atcg::ref_ptr<DielectricMaterial>
MaterialSerializer<DielectricMaterial>::deserialize(const std::filesystem::path& path,
                                                    const nlohmann::json& material_node)
{
    atcg::ref_ptr<DielectricMaterial> material = atcg::make_ref<DielectricMaterial>();

    // Diffuse
    if(material_node.contains(DIFFUSE_KEY))
    {
        std::vector<float> diffuse_color = material_node[DIFFUSE_KEY];
        if(diffuse_color.size() == 3)
        {
            material->setDiffuseColor(glm::vec4(glm::make_vec3(diffuse_color.data()), 1.0f));
        }
        else if(diffuse_color.size() == 4)
        {
            material->setDiffuseColor(glm::make_vec4(diffuse_color.data()));
        }
    }
    else if(material_node.contains(DIFFUSE_TEXTURE_KEY))
    {
        std::filesystem::path diffuse_path = path.parent_path() / material_node[DIFFUSE_TEXTURE_KEY];
        auto img                           = IO::imread(diffuse_path.generic_string(), 2.2f);
        auto diffuse_texture               = atcg::Texture2D::create(img);
        material->setDiffuseTexture(diffuse_texture);
    }

    // Roughness
    if(material_node.contains(ROUGHNESS_KEY))
    {
        float roughness = material_node[ROUGHNESS_KEY];
        material->setRoughness(roughness);
    }
    else if(material_node.contains(ROUGHNESS_TEXTURE_KEY))
    {
        std::filesystem::path roughness_path = path.parent_path() / material_node[ROUGHNESS_TEXTURE_KEY];
        auto img                             = IO::imread(roughness_path.generic_string());
        auto roughness_texture               = atcg::Texture2D::create(img);
        material->setRoughnessTexture(roughness_texture);
    }

    // IoR
    if(material_node.contains(IOR_KEY))
    {
        float ior = material_node[IOR_KEY];
        material->setIor(ior);
    }
    else if(material_node.contains(IOR_TEXTURE_KEY))
    {
        std::filesystem::path ior_path = path.parent_path() / material_node[IOR_TEXTURE_KEY];
        auto img                       = IO::imread(ior_path.generic_string());
        auto ior_texture               = atcg::Texture2D::create(img);
        material->setIorTexture(ior_texture);
    }

    return material;
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

#pragma endregion

}    // namespace atcg