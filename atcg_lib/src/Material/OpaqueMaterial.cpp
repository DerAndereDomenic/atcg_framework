#include <Material/OpaqueMaterial.h>
#include <Renderer/GraphicsAPI.h>
#include <Renderer/Renderer.h>
#include <Core/Application.h>
#include <Material/OpaqueMaterialData.h>

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

namespace atcg
{

class OpaqueMaterial::Impl
{
public:
    Impl(const atcg::Dictionary& dict);

    ~Impl();

    // Real-time data
    atcg::ref_ptr<atcg::Texture2D> _normal_texture;
    atcg::ref_ptr<atcg::Texture2D> _metallic_texture;

    // Optix Data
    atcg::ref_ptr<atcg::Texture2D> _diffuse_texture_gpu;
    atcg::ref_ptr<atcg::Texture2D> _roughness_texture_gpu;
    atcg::ref_ptr<atcg::Texture2D> _metallic_texture_gpu;

    atcg::dref_ptr<OpaqueMaterialData> _material_data;
};

OpaqueMaterial::Impl::Impl(const atcg::Dictionary& dict)
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

    _normal_texture   = dict.getValueOr<atcg::ref_ptr<atcg::Texture2D>>("normal_texture", _normal_texture);
    _metallic_texture = dict.getValueOr<atcg::ref_ptr<atcg::Texture2D>>("metallic_texture", _metallic_texture);
}

OpaqueMaterial::Impl::~Impl()
{
    // Destructor
}

OpaqueMaterial::OpaqueMaterial(const atcg::Dictionary& dict) : MicrofacetMaterial("Opaque", dict)
{
    impl = std::make_unique<Impl>(dict);

    _flags = MaterialFlag::GlossyReflection | MaterialFlag::DiffuseReflection;
}

OpaqueMaterial::~OpaqueMaterial()
{
    // Destructor
}

void OpaqueMaterial::setMetallic(const float metallic)
{
    TextureSpecification spec_metallic;
    spec_metallic.width     = 1;
    spec_metallic.height    = 1;
    spec_metallic.format    = TextureFormat::RFLOAT;
    impl->_metallic_texture = atcg::Texture2D::create(&metallic, spec_metallic);
}

atcg::ref_ptr<atcg::Texture2D> OpaqueMaterial::getNormalTexture() const
{
    return impl->_normal_texture;
}

atcg::ref_ptr<atcg::Texture2D> OpaqueMaterial::getMetallicTexture() const
{
    return impl->_metallic_texture;
}

void OpaqueMaterial::setNormalTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) {}

void OpaqueMaterial::setMetallicTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) {}

void OpaqueMaterial::removeNormalMap()
{
    TextureSpecification spec_normal;
    spec_normal.width  = 1;
    spec_normal.height = 1;
    glm::u8vec4 normal(127, 127, 255, 255);
    impl->_normal_texture = atcg::Texture2D::create(&normal, spec_normal);
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
    atcg::Dictionary dict;
    atcg::ref_ptr<OpaqueMaterial> material = atcg::make_ref<OpaqueMaterial>(dict);

    material->setDiffuseTexture(std::dynamic_pointer_cast<atcg::Texture2D>(getDiffuseTexture()->clone()));
    material->setRoughnessTexture(std::dynamic_pointer_cast<atcg::Texture2D>(getRoughnessTexture()->clone()));
    material->setIorTexture(std::dynamic_pointer_cast<atcg::Texture2D>(getIorTexture()->clone()));
    material->setMetallicTexture(std::dynamic_pointer_cast<atcg::Texture2D>(getMetallicTexture()->clone()));
    material->setNormalTexture(std::dynamic_pointer_cast<atcg::Texture2D>(getNormalTexture()->clone()));

    return material;
}


void OpaqueMaterial::updateData()
{
    impl->_diffuse_texture_gpu   = std::dynamic_pointer_cast<Texture2D>(getDiffuseTexture()->clone());
    impl->_metallic_texture_gpu  = std::dynamic_pointer_cast<Texture2D>(getMetallicTexture()->clone());
    impl->_roughness_texture_gpu = std::dynamic_pointer_cast<Texture2D>(getRoughnessTexture()->clone());

    OpaqueMaterialData data;

    data.diffuse_texture.texture_data.texture   = impl->_diffuse_texture_gpu->getTextureObject();
    data.diffuse_texture.spec                   = impl->_diffuse_texture_gpu->getSpecification();
    data.metallic_texture.texture_data.texture  = impl->_metallic_texture_gpu->getTextureObject();
    data.metallic_texture.spec                  = impl->_metallic_texture_gpu->getSpecification();
    data.roughness_texture.texture_data.texture = impl->_roughness_texture_gpu->getTextureObject();
    data.roughness_texture.spec                 = impl->_roughness_texture_gpu->getSpecification();

    impl->_material_data.upload(&data);
}


void OpaqueMaterial::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                        const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    updateData();

    const std::string ptx_bsdf_filename = "./bin/PBRBSDF_ptx.ptx";
    auto sample_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_pbrbsdf"});
    auto eval_prog_group   = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_pbrbsdf"});
    uint32_t sample_idx    = sbt->addCallableEntry(sample_prog_group, impl->_material_data.get());
    uint32_t eval_idx      = sbt->addCallableEntry(eval_prog_group, impl->_material_data.get());

    BSDFVPtrTable table;
    table.sampleCallIndex = sample_idx;
    table.evalCallIndex   = eval_idx;
    table.flags           = _flags;

    _bsdf_vptr_table.upload(&table);

    markInitialized();
}

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

namespace detail
{
ATCG_INLINE static std::string
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
    atcg::ref_ptr<OpaqueMaterial> material = atcg::make_ref<OpaqueMaterial>(atcg::Dictionary());

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

void OpaqueMaterial::registerMaterial(MaterialRegistry::Registry* registry)
{
    ATCG_REGISTER_MATERIAL(registry, "Opaque", OpaqueMaterial);
}
}    // namespace atcg