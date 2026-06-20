#include <Asset/AssetPanel.h>

#ifndef ATCG_HEADLESS
    #include <implot.h>
#endif

#include <Core/Application.h>
#include <Core/glm.h>
#include <Asset/AssetManagerSystem.h>
#include <portable-file-dialogs.h>
#include <Scene/ComponentGUIHandler.h>
#include <Asset/Project.h>
#include <Scene/Scene.h>
#include <Core/Path.h>
#include <Utils/Utils.h>
#include <Renderer/Renderer.h>

namespace atcg
{
namespace GUI
{

namespace detail
{
void displayOpaqueMaterial(const atcg::ref_ptr<OpaqueMaterial>& material, const std::string& key, bool& updated)
{
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
        }
    }
}

void displayDielectricMaterial(const atcg::ref_ptr<DielectricMaterial>& material, const std::string& key, bool& updated)
{
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
        }
    }
}

void displayNullMaterial(const atcg::ref_ptr<NullMaterial>& material, const std::string& key, bool& updated) {}
}    // namespace detail

AssetPanel::AssetPanel()
{
    {
        auto img     = atcg::IO::imread((atcg::resource_directory() / "folder_icon.png").string());
        _folder_icon = atcg::Texture2D::create(img);
    }

    {
        auto img     = atcg::IO::imread((atcg::resource_directory() / "script_icon.png").string());
        _script_icon = atcg::Texture2D::create(img);
    }

    {
        auto img       = atcg::IO::imread((atcg::resource_directory() / "material_icon.png").string());
        _material_icon = atcg::Texture2D::create(img);
    }

    {
        auto img   = atcg::IO::imread((atcg::resource_directory() / "mesh_icon.png").string());
        _mesh_icon = atcg::Texture2D::create(img);
    }

    {
        auto img    = atcg::IO::imread((atcg::resource_directory() / "image_icon.png").string());
        _image_icon = atcg::Texture2D::create(img);
    }

    {
        _preview_framebuffer = atcg::make_ref<Framebuffer>(512, 512);
        _preview_framebuffer->attachColor();
        _preview_framebuffer->attachDepth();
        _preview_framebuffer->complete();
    }
}

void AssetPanel::displayMaterial(AssetHandle handle)
{
#ifndef ATCG_HEADLESS
    const std::string key = "material";
    auto material_        = AssetManager::getAsset<Material>(handle);

    if(!material_) return;

    bool updated = false;

    float content_scale = atcg::Application::get()->getWindow()->getContentScale();
    ImGui::Separator();

    ImGui::Text("Material");

    int currentIndex = static_cast<int>(material_->getMaterialType());

    constexpr const char* materialTypeLabels[] = {"Opaque", "Dielectric", "Null"};

    atcg::ref_ptr<Material> material = material_->clone();
    material->handle                 = material_->handle;
    if(ImGui::BeginCombo("Material Type", materialTypeToString(material_->getMaterialType())))
    {
        for(int i = 0; i < IM_ARRAYSIZE(materialTypeLabels); ++i)
        {
            bool isSelected = (i == currentIndex);
            if(ImGui::Selectable(materialTypeLabels[i], isSelected))
            {
                auto new_type = static_cast<MaterialType>(i);
                switch(new_type)
                {
                    case MaterialType::MATERIAL_TYPE_OPAQUE:
                        material         = atcg::make_ref<OpaqueMaterial>();
                        material->handle = handle;
                        break;
                    case MaterialType::MATERIAL_TYPE_DIELECTRIC:
                        material         = atcg::make_ref<DielectricMaterial>();
                        material->handle = handle;
                        break;
                    case MaterialType::MATERIAL_TYPE_NULL:
                        material         = atcg::make_ref<NullMaterial>();
                        material->handle = handle;
                        break;
                }
                updated = true;
            }
            if(isSelected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    switch(material->getMaterialType())
    {
        case MaterialType::MATERIAL_TYPE_OPAQUE:
            detail::displayOpaqueMaterial(std::dynamic_pointer_cast<OpaqueMaterial>(material), key, updated);
            break;
        case MaterialType::MATERIAL_TYPE_DIELECTRIC:
            detail::displayDielectricMaterial(std::dynamic_pointer_cast<DielectricMaterial>(material), key, updated);
            break;
        case MaterialType::MATERIAL_TYPE_NULL:
            detail::displayNullMaterial(std::dynamic_pointer_cast<NullMaterial>(material), key, updated);
            break;
    }

    if(updated)
    {
        atcg::RevisionStack::startRecording<AssetEditedRevision>(material_->handle);
        AssetManager::registerAsset(material, AssetManager::getMetaData(material_->handle).name);
        atcg::RevisionStack::endRecording();
    }
#endif
}

void AssetPanel::displayGraph(AssetHandle handle)
{
#ifndef ATCG_HEADLESS
    auto graph     = atcg::AssetManager::getAsset<Graph>(handle);
    int n_vertices = graph ? graph->n_vertices() : 0;
    int n_faces    = graph ? graph->n_faces() : 0;
    if(graph) ImGui::Text(("Type: " + graphTypeToString(graph->type())).c_str());
    ImGui::Text(("Vertices: " + std::to_string(n_vertices)).c_str());
    ImGui::Text(("Faces: " + std::to_string(n_faces)).c_str());
    if(ImGui::Button("Import Mesh##GeometryComponent"))
    {
        auto f =
            pfd::open_file("Choose files to read", pfd::path::home(), {"Obj Files (.obj)", "*.obj"}, pfd::opt::none);
        auto files = f.result();
        if(!files.empty())
        {
            auto mesh    = IO::read_any(files[0]);
            mesh->handle = handle;

            atcg::RevisionStack::startRecording<AssetEditedRevision>(mesh->handle);
            AssetManager::registerAsset(mesh, AssetManager::getMetaData(mesh->handle).name);
            atcg::RevisionStack::endRecording();
        }
    }

    if(!graph || graph->type() != GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH) return;

    // Thumbnail preview
    float content_scale = atcg::Application::get()->getWindow()->getContentScale();
    ImGui::Image((ImTextureID)_preview_framebuffer->getColorAttachement()->getID(),
                 ImVec2(content_scale * 256, content_scale * 256),
                 ImVec2 {0, 1},
                 ImVec2 {1, 0});

    atcg::BoundingBox bbox = graph->getBoundingBox();

    // Place a camera that has a good view on the mesh based on the bounding box
    glm::vec3 center    = bbox.min + (bbox.max - bbox.min) * 0.5f;
    float radius        = glm::length(bbox.max - bbox.min) * 0.5f;
    glm::vec3 direction = glm::vec3(glm::cos(glm::radians(30.0f) * glm::cos(glm::radians(45.0f))),
                                    glm::sin(glm::radians(30.0f)),
                                    glm::cos(glm::radians(30.0f) * glm::sin(glm::radians(45.0f))));
    glm::vec3 cam_pos   = glm::vec3(center.x, center.y, center.z) + 2.0f * radius * direction;
    glm::mat4 view      = glm::lookAt(cam_pos, center, glm::vec3(0, 1, 0));
    glm::mat4 proj      = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, radius * 10.0f);
    glm::mat4 mvp       = proj * view;

    atcg::CameraExtrinsics extrinsics(view);
    atcg::CameraIntrinsics intrinsics(proj);
    atcg::ref_ptr<PerspectiveCamera> camera = atcg::make_ref<PerspectiveCamera>(extrinsics, intrinsics);

    atcg::GraphicsPipeline pipeline =
        atcg::GraphicsPipeline()
            .setShader(atcg::ShaderManager::getShader("mesh_preview"))
            .setRasterizerState(
                atcg::RasterizerState().setCullMode(CullMode::ATCG_BACK_FACE_CULLING).enableCulling(true));

    atcg::GraphicsCommand::beginRenderPass(_preview_framebuffer);

    atcg::GraphicsCommand::clear();
    atcg::Renderer::drawVAO(graph->getVerticesArray(), camera, glm::mat4(1.0f), pipeline, graph->n_vertices());

    atcg::GraphicsCommand::endRenderPass();

#endif
}

void AssetPanel::displayScript(AssetHandle handle)
{
#ifndef ATCG_HEADLESS

    auto current_script = AssetManager::getAsset<PythonScript>(handle);

    if(ImGui::Button("Load Script"))
    {
        auto f =
            pfd::open_file("Choose files to read", pfd::path::home(), {"Python Files (.py)", "*.py"}, pfd::opt::none);
        auto files = f.result();
        if(!files.empty())
        {
            auto script    = atcg::make_ref<atcg::PythonScript>(files[0]);
            script->handle = handle;

            atcg::RevisionStack::startRecording<AssetEditedRevision>(script->handle);
            script->init();
            AssetManager::registerAsset(script, AssetManager::getMetaData(script->handle).name);
            atcg::RevisionStack::endRecording();
        }
    }

    if(current_script)
    {
        std::string source = current_script->getSource();
        ImGui::TextUnformatted(source.c_str());
    }
#endif
}

void AssetPanel::displayShader(AssetHandle handle)
{
#ifndef ATCG_HEADLESS
    auto shader = AssetManager::getAsset<Shader>(handle);

    // if(shader)
    // {
    //     if(shader->isComputeShader())
    //     {
    //         _current_compute_path = shader->getComputePath();
    //     }
    //     else
    //     {
    //         if(shader->hasGeometryShader())
    //         {
    //             _current_geometry_path = shader->getComputePath();
    //         }

    //         _current_vertex_path   = shader->getVertexPath();
    //         _current_fragment_path = shader->getFragmentPath();
    //     }
    // }

    if(ImGui::Button("Load Vertex Shader"))
    {
        auto f =
            pfd::open_file("Choose files to read", pfd::path::home(), {"Vertex Shader (.vs)", "*.vs"}, pfd::opt::none);
        auto files = f.result();
        if(!files.empty())
        {
            _current_vertex_path = files[0];
        }
    }
    if(_current_vertex_path != "")
    {
        ImGui::Text(_current_vertex_path.c_str());
    }

    if(ImGui::Button("Load Fragment Shader"))
    {
        auto f     = pfd::open_file("Choose files to read",
                                    pfd::path::home(),
                                    {"Fragment Shader (.fs)", "*.fs"},
                                    pfd::opt::none);
        auto files = f.result();
        if(!files.empty())
        {
            _current_fragment_path = files[0];
        }
    }
    if(_current_fragment_path != "")
    {
        ImGui::Text(_current_fragment_path.c_str());
    }

    if(ImGui::Button("Load Geometry Shader"))
    {
        auto f     = pfd::open_file("Choose files to read",
                                    pfd::path::home(),
                                    {"Geometry Shader (.gs)", "*.gs"},
                                    pfd::opt::none);
        auto files = f.result();
        if(!files.empty())
        {
            _current_geometry_path = files[0];
        }
    }
    if(_current_geometry_path != "")
    {
        ImGui::Text(_current_geometry_path.c_str());
    }

    if(ImGui::Button("Load Compute Shader"))
    {
        auto f     = pfd::open_file("Choose files to read",
                                    pfd::path::home(),
                                    {"Compute Shader (.glsl)", "*.glsl"},
                                    pfd::opt::none);
        auto files = f.result();
        if(!files.empty())
        {
            _current_compute_path = files[0];
        }
    }
    if(_current_compute_path != "")
    {
        ImGui::Text(_current_compute_path.c_str());
    }

    if(ImGui::Button("Compile"))
    {
        if(_current_compute_path != "")
        {
            shader = atcg::make_ref<Shader>(_current_compute_path);
        }
        else if(_current_vertex_path != "" && _current_fragment_path != "")
        {
            if(_current_geometry_path != "")
            {
                shader = atcg::make_ref<Shader>(_current_vertex_path, _current_fragment_path, _current_geometry_path);
            }
            else
            {
                shader = atcg::make_ref<Shader>(_current_vertex_path, _current_fragment_path);
            }
        }
        else
        {
            shader = nullptr;
        }

        if(shader)
        {
            shader->handle = handle;

            atcg::RevisionStack::startRecording<AssetEditedRevision>(shader->handle);
            AssetManager::registerAsset(shader, AssetManager::getMetaData(shader->handle).name);
            atcg::RevisionStack::endRecording();
        }
    }

    if(shader)
    {
        if(shader->isComputeShader())
        {
            auto cs_source = shader->getSource(atcg::ShaderType::COMPUTE);

            ImGui::Text("Compute Shader:");
            ImGui::TextUnformatted(cs_source.c_str());
            ImGui::Separator();
        }
        else
        {
            {
                auto vs_source = shader->getSource(atcg::ShaderType::VERTEX);

                ImGui::Text("Vertex Shader:");
                ImGui::TextUnformatted(vs_source.c_str());
                ImGui::Separator();
            }

            {
                auto fs_source = shader->getSource(atcg::ShaderType::FRAGMENT);

                ImGui::Text("Fragment Shader:");
                ImGui::TextUnformatted(fs_source.c_str());
                ImGui::Separator();
            }

            if(shader->hasGeometryShader())
            {
                auto gs_source = shader->getSource(atcg::ShaderType::GEOMETRY);

                ImGui::Text("Geometry Shader:");
                ImGui::TextUnformatted(gs_source.c_str());
            }
        }
    }
    else
    {
        ImGui::Text("Invalid Shader");
    }

#endif
}

void AssetPanel::displayTexture2D(AssetHandle handle)
{
#ifndef ATCG_HEADLESS
    if(ImGui::Button(("Load Image##tex2dasset")))
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

            texture->handle = handle;

            atcg::RevisionStack::startRecording<AssetEditedRevision>(texture->handle);
            AssetManager::registerAsset(texture, AssetManager::getMetaData(texture->handle).name);
            atcg::RevisionStack::endRecording();
        }
    }

    auto texture = AssetManager::getAsset<Texture2D>(handle);

    if(texture)
    {
        float content_scale = atcg::Application::get()->getWindow()->getContentScale();
        float aspect_ratio  = (float)texture->width() / (float)texture->height();
        ImGui::Image((ImTextureID)texture->getID(),
                     ImVec2(content_scale * 256, content_scale * 256 / aspect_ratio),
                     ImVec2 {0, 1},
                     ImVec2 {1, 0});
    }

#endif
}

void AssetPanel::displayTexture3D(AssetHandle handle)
{
#ifndef ATCG_HEADLESS
    auto texture = AssetManager::getAsset<Texture3D>(handle);


    ImGui::InputInt("Width##Texture3D", (int*)&_spec_3d.width);
    ImGui::InputInt("Height##Texture3D", (int*)&_spec_3d.height);
    ImGui::InputInt("Depth##Texture3D", (int*)&_spec_3d.depth);

    if(ImGui::BeginCombo("Select Format##Texture3D", textureFormatToString(_spec_3d.format)))
    {
        for(int i = 0; i < (int)TextureFormat::_NUM_FORMATS; ++i)
        {
            bool is_selected = (int)_spec_3d.format == i;

            if(ImGui::Selectable((std::string(textureFormatToString((TextureFormat)i)) + "##Texture3D").c_str(),
                                 is_selected))
            {
                _spec_3d.format = (TextureFormat)i;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        ImGui::EndCombo();
    }

    if(ImGui::Button("Path##Texture3D"))
    {
        auto f     = pfd::open_file("Choose files to read",
                                    pfd::path::home(),
                                    {"Compute Shader (.bin)", "*.bin"},
                                    pfd::opt::none);
        auto files = f.result();
        if(!files.empty())
        {
            _current_texture_3d_path = files[0];
        }
    }

    if(_current_texture_3d_path != "")
    {
        ImGui::Text(_current_texture_3d_path.c_str());
    }

    ImGui::BeginDisabled(_current_texture_3d_path == "");
    if(ImGui::Button("Load##Texture3D"))
    {
        std::ifstream summary_file(_current_texture_3d_path, std::ios::in | std::ios::binary);
        std::vector<uint8_t> buffer_char(std::istreambuf_iterator<char>(summary_file), {});
        summary_file.close();

        auto texture    = Texture3D::create((const char*)buffer_char.data(), _spec_3d);
        texture->handle = handle;

        atcg::RevisionStack::startRecording<AssetEditedRevision>(texture->handle);
        AssetManager::registerAsset(texture, AssetManager::getMetaData(texture->handle).name);
        atcg::RevisionStack::endRecording();

        _spec_3d                 = TextureSpecification();
        _current_texture_3d_path = "";
    }
    ImGui::EndDisabled();


    if(texture)
    {
        auto spec = texture->getSpecification();

        if(!_preview || _preview->width() != spec.width || _preview->height() != spec.height)
        {
            _preview = Texture2D::create(spec);
        }

        auto slice = texture->getData(atcg::GPU)[_slice];

        if(ImGui::SliderInt("Slice##Texture3D", (int*)&_slice, 0, spec.depth - 1))
        {
            _preview->setData(slice);
        }

        float content_scale = atcg::Application::get()->getWindow()->getContentScale();
        float aspect_ratio  = (float)texture->width() / (float)texture->height();
        ImGui::Image((ImTextureID)_preview->getID(),
                     ImVec2(content_scale * 256, content_scale * 256 / aspect_ratio),
                     ImVec2 {0, 1},
                     ImVec2 {1, 0});
    }
#endif
}

void AssetPanel::displayScene(AssetHandle handle)
{
#ifndef ATCG_HEADLESS
    auto scene = AssetManager::getAsset<Scene>(handle);

    float content_scale = atcg::Application::get()->getWindow()->getContentScale();

    AssetHandle skybox_handle = 0;

    if(scene->hasSkybox())
    {
        skybox_handle = scene->getSkyboxTexture()->handle;
    }

    ImGui::Text("Skybox:");
    auto new_handle = Utils::displayTexture2DSelection("skybox", skybox_handle);
    bool updated    = (new_handle != skybox_handle);

    if(updated)
    {
        if(AssetManager::isAssetHandleValid(new_handle))
        {
            auto skybox_texture = AssetManager::getAsset<Texture2D>(new_handle);
            scene->setSkybox(skybox_texture);
        }
        else
        {
            scene->removeSkybox();
        }
    }


    auto scene_camera = scene->getCamera();
    if(scene_camera)
    {
        uint32_t width                    = atcg::Renderer::getFramebuffer()->width();
        uint32_t height                   = atcg::Renderer::getFramebuffer()->height();
        atcg::CameraIntrinsics intrinsics = scene_camera->getIntrinsics();
        glm::mat3 K                       = atcg::CameraUtils::convert_to_opencv(intrinsics, width, height);
        uint32_t id                       = (uint32_t)typeid(atcg::Camera).hash_code();
        bool updated                      = false;

        float fx = K[0][0];
        float fy = K[1][1];
        float cx = K[0][2];
        float cy = K[1][2];

        float f[2]      = {fx, fy};
        float c[2]      = {cx, cy};
        float offset[2] = {intrinsics.opticalCenter().x, intrinsics.opticalCenter().y};

        float aspect_ratio = intrinsics.aspectRatio();
        float fov          = intrinsics.FOV();

        std::stringstream label;
        label << "Aspect##" << id;
        if(ImGui::DragFloat(label.str().c_str(), &aspect_ratio, 0.05f, 0.1f, 5.0f))
        {
            intrinsics.setAspectRatio(aspect_ratio);
            updated = true;
        }

        ImGui::SameLine();
        if(ImGui::Button("Reset"))
        {
            intrinsics.setAspectRatio(float(width) / float(height));
            updated = true;
        }

        label.str(std::string());
        label << "FOV##" << id;
        if(ImGui::DragFloat(label.str().c_str(), &fov, 0.5f, 10.0f, 120.0f))
        {
            intrinsics.setFOV(fov);
            updated = true;
        }

        label.str(std::string());
        label << "Optical Center##" << id;
        if(ImGui::DragFloat2(label.str().c_str(), offset, 0.01f, -1.0f, 1.0f))
        {
            intrinsics.setOpticalCenter(glm::make_vec2(offset));
            updated = true;
        }

        ImGui::Separator();

        label.str(std::string());
        label << "Focal Length##" << id;
        if(ImGui::DragFloat2(label.str().c_str(), f, 0.5f, 1.0f, 4096.0f))
        {
            intrinsics = atcg::CameraUtils::convert_from_opencv(f[0],
                                                                f[1],
                                                                c[0],
                                                                c[1],
                                                                intrinsics.zNear(),
                                                                intrinsics.zFar(),
                                                                width,
                                                                height);
            updated    = true;
        }

        label.str(std::string());
        label << "Principal Point##" << id;
        if(ImGui::DragFloat2(label.str().c_str(), c, 0.5f, 1.0f, 4096.0f))
        {
            intrinsics = atcg::CameraUtils::convert_from_opencv(f[0],
                                                                f[1],
                                                                c[0],
                                                                c[1],
                                                                intrinsics.zNear(),
                                                                intrinsics.zFar(),
                                                                width,
                                                                height);
            updated    = true;
        }

        float exposure = intrinsics.getExposure();
        if(ImGui::DragFloat("Exposure##Camera", &exposure, 0.01f, 0.0f, 10.0f))
        {
            intrinsics.setExposure(exposure);
            updated = true;
        }

        if(updated) scene_camera->setIntrinsics(intrinsics);
    }

    ImGui::Separator();

    if(ImGui::Button("Make active"))
    {
        Project::getActive()->setActiveScene(handle);
    }

#endif
}

#ifndef ATCG_HEADLESS
ATCG_INLINE static bool ImageTextButton(ImTextureID textureID,
                                        const char* label,
                                        ImVec2 imageSize,
                                        float spacing  = 4.0f,
                                        ImVec2 padding = ImVec2(4, 4))
{
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    ImVec2 textSize      = ImGui::CalcTextSize(label);

    // compute button size
    float buttonWidth  = std::max(imageSize.x, textSize.x) + padding.x * 2.0f;
    float buttonHeight = imageSize.y + spacing + textSize.y + padding.y * 2.0f;
    ImVec2 buttonSize(buttonWidth, buttonHeight);

    // position
    ImVec2 pos = ImGui::GetCursorScreenPos();

    // handle button interaction
    bool clicked = ImGui::InvisibleButton(label, buttonSize);

    // Draw background highlight based on state
    ImU32 col = 0;
    if(ImGui::IsItemActive())
        col = ImGui::GetColorU32(ImGuiCol_ButtonActive);
    else if(ImGui::IsItemHovered())
        col = ImGui::GetColorU32(ImGuiCol_ButtonHovered);

    if(col != 0)
    {
        drawList->AddRectFilled(pos,
                                ImVec2(pos.x + buttonSize.x, pos.y + buttonSize.y),
                                col,
                                ImGui::GetStyle().FrameRounding);
    }

    // draw image
    ImVec2 imagePos = ImVec2(pos.x + (buttonWidth - imageSize.x) * 0.5f, pos.y + padding.y);
    drawList->AddImage(textureID,
                       imagePos,
                       ImVec2(imagePos.x + imageSize.x, imagePos.y + imageSize.y),
                       ImVec2 {0, 1},
                       ImVec2 {1, 0});

    // draw text
    ImVec2 textPos = ImVec2(pos.x + (buttonWidth - textSize.x) * 0.5f, imagePos.y + imageSize.y + spacing);
    drawList->AddText(textPos, ImGui::GetColorU32(ImGuiCol_Text), label);

    return clicked;
}
#endif

void AssetPanel::drawAssetList()
{
#ifndef ATCG_HEADLESS
    // Begin a scrollable horizontal region
    ImGui::BeginChild("AssetListHorizontal", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

    const auto& registry = AssetManager::getAssetRegistry();

    float content_scale = atcg::Application::get()->getWindow()->getContentScale();
    if(_panel_state == AssetType::None)
    {
        if(ImageTextButton((ImTextureID)_folder_icon->getID(),
                           "Scenes",
                           ImVec2(content_scale * 64, content_scale * 64)))
        {
            _panel_state = AssetType::Scene;
        }
        ImGui::SameLine();
        if(ImageTextButton((ImTextureID)_folder_icon->getID(),
                           "Textures",
                           ImVec2(content_scale * 64, content_scale * 64)))
        {
            _panel_state = AssetType::Texture2D;
        }
        ImGui::SameLine();
        if(ImageTextButton((ImTextureID)_folder_icon->getID(),
                           "3D Textures",
                           ImVec2(content_scale * 64, content_scale * 64)))
        {
            _panel_state = AssetType::Texture3D;
        }
        ImGui::SameLine();
        if(ImageTextButton((ImTextureID)_folder_icon->getID(),
                           "Materials",
                           ImVec2(content_scale * 64, content_scale * 64)))
        {
            _panel_state = AssetType::Material;
        }
        ImGui::SameLine();
        if(ImageTextButton((ImTextureID)_folder_icon->getID(),
                           "Models",
                           ImVec2(content_scale * 64, content_scale * 64)))
        {
            _panel_state = AssetType::Graph;
        }
        ImGui::SameLine();
        if(ImageTextButton((ImTextureID)_folder_icon->getID(),
                           "Scripts",
                           ImVec2(content_scale * 64, content_scale * 64)))
        {
            _panel_state = AssetType::Script;
        }
        ImGui::SameLine();
        if(ImageTextButton((ImTextureID)_folder_icon->getID(),
                           "Shader",
                           ImVec2(content_scale * 64, content_scale * 64)))
        {
            _panel_state = AssetType::Shader;
        }
        ImGui::SameLine();
    }
    else
    {
        if(ImGui::Button("Back"))
        {
            _panel_state = AssetType::None;
        }

        ImGui::SameLine();

        drawAdd();
    }

    for(auto entry: registry)
    {
        const auto& data = entry.second;

        if(_panel_state == data.type)
        {
            auto handle     = entry.first;
            std::string tag = data.name;

            bool isSelected = (handle == _selected_handle);
            if(isSelected) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.9f, 1.0f));

            auto icon = _script_icon;

            if(data.type == AssetType::Material)
            {
                icon = _material_icon;
            }
            else if(data.type == AssetType::Graph)
            {
                icon = _mesh_icon;
            }
            else if(data.type == AssetType::Texture2D || data.type == AssetType::Texture3D)
            {
                icon = _image_icon;
            }

            ImGui::PushID(handle);
            if(ImageTextButton((ImTextureID)icon->getID(), tag.c_str(), ImVec2(content_scale * 96, content_scale * 96)))
            {
                selectAsset(handle);
            }
            ImGui::PopID();

            if(isSelected) ImGui::PopStyleColor();    // always pop if you pushed

            ImGui::SameLine();
        }
    }

    ImGui::EndChild();
#endif
}

void AssetPanel::drawAdd()
{
#ifndef ATCG_HEADLESS

    if(ImGui::Button("Create new Asset"))
    {
        AssetHandle new_asset = 0;
        if(_panel_state == AssetType::Graph)
        {
            AssetMetaData data;
            data.type = AssetType::Graph;
            data.name = "graph";
            new_asset = AssetManager::registerAsset(data);
        }
        if(_panel_state == AssetType::Material)
        {
            new_asset = AssetManager::registerAsset(atcg::make_ref<OpaqueMaterial>(), "material");
        }
        if(_panel_state == AssetType::Script)
        {
            AssetMetaData data;
            data.type = AssetType::Script;
            data.name = "script";
            new_asset = AssetManager::registerAsset(data);
        }
        if(_panel_state == AssetType::Shader)
        {
            AssetMetaData data;
            data.type = AssetType::Shader;
            data.name = "shader";
            new_asset = AssetManager::registerAsset(data);
        }
        if(_panel_state == AssetType::Texture2D)
        {
            AssetMetaData data;
            data.type = AssetType::Texture2D;
            data.name = "texture";
            new_asset = AssetManager::registerAsset(data);
        }
        if(_panel_state == AssetType::Texture3D)
        {
            AssetMetaData data;
            data.type = AssetType::Texture3D;
            data.name = "texture3d";
            new_asset = AssetManager::registerAsset(data);
        }
        if(_panel_state == AssetType::Scene)
        {
            new_asset = AssetManager::registerAsset(atcg::make_ref<Scene>(), "scene");
        }

        if(AssetManager::isAssetHandleValid(new_asset))
        {
            atcg::RevisionStack::startRecording<AssetAddedRevision>(new_asset);
            atcg::RevisionStack::endRecording();

            // Directly select created asset
            selectAsset(new_asset);
        }
    }
#endif
}

void AssetPanel::drawAssetPanel()
{
#ifndef ATCG_HEADLESS
    ImGui::Begin("Assets");

    if(ImGui::IsMouseDown(0) && ImGui::IsWindowHovered() && !ImGui::IsAnyItemHovered() && !ImGui::IsAnyItemActive())
    {
        selectAsset(0);
    }

    drawAssetList();

    ImGui::End();
#endif
}

void AssetPanel::drawAssetEditor()
{
#ifndef ATCG_HEADLESS
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
            atcg::RevisionStack::startRecording<AssetEditedRevision>(_selected_handle);
            AssetManager::updateName(_selected_handle, std::string(buffer));
            atcg::RevisionStack::endRecording();
        }

        if(data.type == AssetType::Material)
        {
            displayMaterial(_selected_handle);
        }
        else if(data.type == AssetType::Graph)
        {
            displayGraph(_selected_handle);
        }
        else if(data.type == AssetType::Script)
        {
            displayScript(_selected_handle);
        }
        else if(data.type == AssetType::Shader)
        {
            displayShader(_selected_handle);
        }
        else if(data.type == AssetType::Texture2D)
        {
            displayTexture2D(_selected_handle);
        }
        else if(data.type == AssetType::Texture3D)
        {
            displayTexture3D(_selected_handle);
        }
        else if(data.type == AssetType::Scene)
        {
            displayScene(_selected_handle);
        }

        ImGui::Separator();

        bool disabled = false;
        auto metadata = AssetManager::getMetaData(_selected_handle);
        if(metadata.type == AssetType::Scene)
        {
            if(Project::getActive()->getActiveScene()->handle == _selected_handle)
            {
                disabled = true;
                ImGui::Text("Can't delete active scene");
            }
        }

        ImGui::BeginDisabled(disabled);
        if(ImGui::Button("Delete"))
        {
            atcg::RevisionStack::startRecording<AssetRemovedRevision>(_selected_handle);
            AssetManager::removeAsset(_selected_handle);
            atcg::RevisionStack::endRecording();

            selectAsset(0);
        }
        ImGui::EndDisabled();
    }

    ImGui::End();
#endif
}

void AssetPanel::renderPanel()
{
#ifndef ATCG_HEADLESS

    drawAssetPanel();

    drawAssetEditor();

#endif
}

void AssetPanel::selectAsset(AssetHandle handle)
{
    _selected_handle = handle;

    // Clean up
    _current_vertex_path   = "";
    _current_fragment_path = "";
    _current_geometry_path = "";
    _current_compute_path  = "";

    _spec_3d                 = TextureSpecification();
    _current_texture_3d_path = "";
}

}    // namespace GUI
}    // namespace atcg