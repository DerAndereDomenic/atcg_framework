#include <Scene/SceneHierarchyPanel.h>

#include <imgui.h>
#include <Scene/Components.h>
#include <portable-file-dialogs.h>
#include <DataStructure/TorchUtils.h>
#include <Core/Application.h>

namespace atcg
{
void SceneHierarchyPanel::drawEntityNode(Entity entity)
{
    auto& tag = entity.getComponent<NameComponent>().name;

    ImGuiTreeNodeFlags flags =
        ((_selected_entity && _selected_entity.getComponent<IDComponent>().ID == entity.getComponent<IDComponent>().ID)
             ? ImGuiTreeNodeFlags_Selected
             : 0) |
        ImGuiTreeNodeFlags_Bullet;
    flags |= ImGuiTreeNodeFlags_SpanAvailWidth;
    bool opened = ImGui::TreeNodeEx((void*)(uint64_t)entity.getComponent<IDComponent>().ID, flags, tag.c_str());
    if(ImGui::IsItemClicked())
    {
        _selected_entity = entity;
    }

    // bool entityDeleted = false;
    // if(ImGui::BeginPopupContextItem())
    // {
    //     if(ImGui::MenuItem("Delete Entity")) entityDeleted = true;

    //     ImGui::EndPopup();
    // }

    if(opened)
    {
        ImGui::TreePop();
    }

    // if(entityDeleted)
    // {
    //     m_Context->DestroyEntity(entity);
    //     if(m_SelectionContext == entity) m_SelectionContext = {};
    // }
}

void SceneHierarchyPanel::drawSceneProperties()
{
    float content_scale                    = atcg::Application::get()->getWindow()->getContentScale();
    const ImGuiTreeNodeFlags treeNodeFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed |
                                             ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap |
                                             ImGuiTreeNodeFlags_FramePadding;

    bool open = ImGui::TreeNodeEx((void*)typeid(atcg::Scene).hash_code(), treeNodeFlags, "Skybox");

    ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();

    if(open)
    {
        if(Renderer::hasSkybox())
        {
            ImGui::Image((void*)(uint64_t)Renderer::getSkyboxTexture()->getID(),
                         ImVec2(content_scale * 128, content_scale * 64),
                         ImVec2 {0, 1},
                         ImVec2 {1, 0});
            if(ImGui::Button("Remove skybox##skybox"))
            {
                Renderer::removeSkybox();
            }
        }
        else
        {
            glm::vec4 clear_color = Renderer::getClearColor();
            if(ImGui::ColorEdit4("Background color#skybox", glm::value_ptr(clear_color)))
            {
                Renderer::setClearColor(clear_color);
            }

            if(ImGui::Button("Add Skybox..."))
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
                    auto img = IO::imread(files[0]);
                    Renderer::setSkybox(img);
                }
            }
        }
        ImGui::TreePop();
    }
}

SceneHierarchyPanel::SceneHierarchyPanel(const atcg::ref_ptr<Scene>& scene)
    : _scene(scene),
      _gui_handler(atcg::make_ref<atcg::ComponentGUIHandler>(scene))
{
}
}    // namespace atcg