#include <Asset/AssetPanel.h>

#ifndef ATCG_HEADLESS
    #include <implot.h>
#endif

#include <Core/Application.h>
#include <Asset/AssetManagerSystem.h>
#include <portable-file-dialogs.h>
#include <Renderer/Material.h>

namespace atcg
{
namespace GUI
{

bool displayMaterial(const std::string& key, const atcg::ref_ptr<Material>& material)
{
    bool updated = false;

    float content_scale = atcg::Application::get()->getWindow()->getContentScale();
    ImGui::Separator();

    ImGui::Text("Material");

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
        }
    }

    return updated;
}

void AssetPanel::drawAssetList()
{
    // Begin a scrollable horizontal region
    ImGui::BeginChild("AssetListHorizontal", ImVec2(0, 80), false, ImGuiWindowFlags_HorizontalScrollbar);

    const auto& registry = AssetManager::getAssetRegistry();

    for(auto entry: registry)
    {
        const auto& data = entry.second;
        auto handle      = entry.first;
        std::string tag  = data.name;

        bool isSelected = (handle == _selected_handle);
        if(isSelected) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.9f, 1.0f));

        ImGui::PushID(handle);
        if(ImGui::Button(tag.c_str()))
        {
            selectAsset(handle);
        }
        ImGui::PopID();

        if(isSelected) ImGui::PopStyleColor();    // always pop if you pushed

        ImGui::SameLine();
    }

    ImGui::EndChild();
}

void AssetPanel::drawAdd()
{
    if(ImGui::Button("Add..."))
    {
        ImGui::OpenPopup("AddPopup");
    }

    if(ImGui::BeginPopup("AddPopup"))
    {
        if(ImGui::MenuItem("Graph"))
        {
            AssetMetaData data;
            data.type = AssetType::Graph;
            data.name = "graph";
            AssetManager::registerAsset(data);
        }
        if(ImGui::MenuItem("Material"))
        {
            AssetManager::registerAsset(atcg::make_ref<Material>(), "material");
        }
        if(ImGui::MenuItem("Script"))
        {
            AssetMetaData data;
            data.type = AssetType::Script;
            data.name = "script";
            AssetManager::registerAsset(data);
        }
        ImGui::EndPopup();
    }
}

void AssetPanel::renderPanel()
{
#ifndef ATCG_HEADLESS

    ImGui::Begin("Assets");

    if(ImGui::IsMouseDown(0) && ImGui::IsWindowHovered() && !ImGui::IsAnyItemHovered() && !ImGui::IsAnyItemActive())
    {
        selectAsset(0);
    }

    drawAdd();
    drawAssetList();

    ImGui::End();

    ImGui::Begin("Asset Editor");

    if(_selected_handle != 0)
    {
        const auto& data = AssetManager::getMetaData(_selected_handle);

        const std::string& tag = data.name;
        char buffer[256];
        memset(buffer, 0, sizeof(buffer));
        // ? strncpy_s not available in gcc. Is this unsafe?
        memcpy(buffer, tag.c_str(), sizeof(buffer));
        if(ImGui::InputText("Name##asset", buffer, sizeof(buffer)))
        {
            AssetManager::updateName(_selected_handle, std::string(buffer));
        }

        if(data.type == AssetType::Material)
        {
            displayMaterial("asset", atcg::AssetManager::getAsset<Material>(_selected_handle));
        }
        else if(data.type == AssetType::Graph)
        {
            auto graph     = atcg::AssetManager::getAsset<Graph>(_selected_handle);
            int n_vertices = graph ? graph->n_vertices() : 0;
            int n_faces    = graph ? graph->n_faces() : 0;
            ImGui::Text(("Vertices: " + std::to_string(n_vertices)).c_str());
            ImGui::Text(("Faces: " + std::to_string(n_faces)).c_str());
            if(ImGui::Button("Import Mesh##GeometryComponent"))
            {
                auto f     = pfd::open_file("Choose files to read",
                                        pfd::path::home(),
                                            {"Obj Files (.obj)", "*.obj"},
                                        pfd::opt::none);
                auto files = f.result();
                if(!files.empty())
                {
                    auto mesh    = IO::read_any(files[0]);
                    mesh->handle = _selected_handle;
                    AssetManager::registerAsset(mesh, AssetManager::getMetaData(mesh->handle));
                }
            }
        }
        else if(data.type == AssetType::Script)
        {
            if(ImGui::Button("Load Script"))
            {
                auto f     = pfd::open_file("Choose files to read",
                                        pfd::path::home(),
                                            {"Python Files (.py)", "*.py"},
                                        pfd::opt::none);
                auto files = f.result();
                if(!files.empty())
                {
                    auto script = atcg::make_ref<atcg::PythonScript>(files[0]);
                    // TODO
                    script->init();
                    // component.script->onAttach();
                    script->handle = _selected_handle;
                    AssetManager::registerAsset(script, AssetManager::getMetaData(script->handle));
                }
            }
        }

        ImGui::Separator();

        if(ImGui::Button("Delete"))
        {
            AssetManager::removeAsset(_selected_handle);

            selectAsset(0);
        }
    }

    ImGui::End();

#endif
}

void AssetPanel::selectAsset(AssetHandle handle)
{
    _selected_handle = handle;
}

}    // namespace GUI
}    // namespace atcg