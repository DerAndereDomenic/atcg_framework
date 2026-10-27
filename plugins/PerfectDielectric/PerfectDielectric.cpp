#include <stdio.h>
#include <Plugin/Plugin.h>

#include "PerfectDielectric.h"

#include <ATCG.h>

#include <Core/Path.h>
#include <Core/Assert.h>
#include <Core/CUDA.h>
#include <Core/Common.h>
#include <Scene/Components.h>
#include <Scene/Entity.h>
#include <DataStructure/WorkerPool.h>
#include <portable-file-dialogs.h>

#ifdef ATCG_CUDA_BACKEND
    #include <Shape/Shape.h>
    #include <Shape/ShapeInstance.h>
    #include <Shape/MeshShape.h>
    #include <Emitter/MeshEmitter.h>
    #include <Scene/SceneAdapter.h>
#endif

#ifndef ATCG_HEADLESS
    #include <imgui.h>
#endif


#define IOR_KEY         "IoR"
#define IOR_TEXTURE_KEY "IoRTexture"
#define TYPE_KEY        "Type"

namespace atcg
{

PerfectDielectric::PerfectDielectric() : atcg::Material("PerfectDielectric")
{
    atcg::TextureSpecification spec_ior;
    spec_ior.width  = 1;
    spec_ior.height = 1;
    spec_ior.format = atcg::TextureFormat::RFLOAT;
    float ior       = 1.5f;
    _ior_texture    = atcg::Texture2D::create(&ior, spec_ior);
}

void PerfectDielectric::uploadMaterial(atcg::RendererSystem* renderer, const atcg::ref_ptr<atcg::Shader>& shader)
{
    ATCG_ASSERT(!_uploaded, "Material was already uploaded");

    uint32_t diffuse_id  = renderer->popTextureID();
    _used_texture_ids[0] = diffuse_id;

    uint32_t normal_id   = renderer->popTextureID();
    _used_texture_ids[1] = normal_id;

    uint32_t roughness_id = renderer->popTextureID();
    _used_texture_ids[2]  = roughness_id;

    uint32_t metallic_id = renderer->popTextureID();
    _used_texture_ids[3] = metallic_id;

    uint32_t ior_id      = renderer->popTextureID();
    _used_texture_ids[4] = ior_id;

    // Select shading functions
    shader->selectSubroutine("sr_eval_brdf", "eval_brdf_glass");
    shader->selectSubroutine("sr_image_based_lighting", "image_based_lighting_glass");

    _uploaded = true;
}

atcg::ref_ptr<atcg::Material> PerfectDielectric::clone() const
{
    atcg::ref_ptr<PerfectDielectric> material = atcg::make_ref<PerfectDielectric>();

    material->setIORTexture(std::dynamic_pointer_cast<atcg::Texture2D>(getIORTexture()->clone()));

    return material;
}

void PerfectDielectric::setIOR(const float& ior)
{
    atcg::TextureSpecification spec_ior;
    spec_ior.width  = 1;
    spec_ior.height = 1;
    spec_ior.format = atcg::TextureFormat::RFLOAT;
    _ior_texture    = atcg::Texture2D::create(&ior, spec_ior);
}

bool MaterialGUIRenderer<PerfectDielectric>::renderGUI(const atcg::ref_ptr<PerfectDielectric>& material,
                                                       const std::string& key,
                                                       bool& deactivated)
{
    bool updated = false;
#ifndef ATCG_HEADLESS
    float content_scale = atcg::Application::get()->getWindow()->getContentScale();

    {
        auto spec        = material->getIORTexture()->getSpecification();
        bool useTextures = spec.width != 1 || spec.height != 1;

        if(!useTextures)
        {
            auto data = material->getIORTexture()->getData(atcg::CPU);
            float ior = data.item<float>();

            if(ImGui::DragFloat(("IoR##" + key).c_str(), &ior, 0.005f, 1.0f, 2.5f))
            {
                material->setIOR(ior);
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
                    material->setIORTexture(texture);
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
                material->setIOR(1.5f);
                updated = true;
            }
            else
                ImGui::Image((ImTextureID)material->getIORTexture()->getID(),
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

void MaterialSerializer<PerfectDielectric>::serialize(const atcg::ref_ptr<PerfectDielectric>& material,
                                                      const std::filesystem::path& path)
{
    nlohmann::json material_json;

    material_json["Version"] = "1.0";

    auto ior_texture     = material->getIORTexture();
    bool use_ior_texture = !(ior_texture->width() == 1 && ior_texture->height() == 1);

    material_json[TYPE_KEY] = material->getMaterialType();

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

atcg::ref_ptr<PerfectDielectric> MaterialSerializer<PerfectDielectric>::deserialize(const std::filesystem::path& path,
                                                                                    const nlohmann::json& material_node)
{
    atcg::ref_ptr<PerfectDielectric> material = atcg::make_ref<PerfectDielectric>();

    // IoR
    if(material_node.contains(IOR_KEY))
    {
        float ior = material_node[IOR_KEY];
        material->setIOR(ior);
    }
    else if(material_node.contains(IOR_TEXTURE_KEY))
    {
        std::filesystem::path ior_path = path.parent_path() / material_node[IOR_TEXTURE_KEY];
        auto img                       = IO::imread(ior_path.generic_string());
        auto ior_texture               = atcg::Texture2D::create(img);
        material->setIORTexture(ior_texture);
    }

    return material;
}

}    // namespace atcg