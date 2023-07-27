#include <Scene/SceneHierarchyPanel.h>

#include <imgui.h>
#include <Scene/Components.h>

namespace atcg
{

void SceneHierarchyPanel::drawEntityNode(Entity entity)
{
    auto& tag = entity.getComponent<NameComponent>().name;

    ImGuiTreeNodeFlags flags =
        ((_selected_entity.getComponent<IDComponent>().ID == entity.getComponent<IDComponent>().ID)
             ? ImGuiTreeNodeFlags_Selected
             : 0) |
        ImGuiTreeNodeFlags_Bullet;
    flags |= ImGuiTreeNodeFlags_SpanAvailWidth;
    bool opened = ImGui::TreeNodeEx((void*)(uint64_t)entity.getComponent<IDComponent>().ID, flags, tag.c_str());
    if(ImGui::IsItemClicked()) { _selected_entity = entity; }

    // bool entityDeleted = false;
    // if(ImGui::BeginPopupContextItem())
    // {
    //     if(ImGui::MenuItem("Delete Entity")) entityDeleted = true;

    //     ImGui::EndPopup();
    // }

    if(opened) { ImGui::TreePop(); }

    // if(entityDeleted)
    // {
    //     m_Context->DestroyEntity(entity);
    //     if(m_SelectionContext == entity) m_SelectionContext = {};
    // }
}


SceneHierarchyPanel::SceneHierarchyPanel(const atcg::ref_ptr<Scene>& scene) : _scene(scene) {}

void SceneHierarchyPanel::renderPanel()
{
    ImGui::Begin("Scene Hierarchy");

    for(auto e: _scene->getAllEntitiesWith<NameComponent>())
    {
        Entity entity(e, _scene.get());
        drawEntityNode(entity);
    }

    ImGui::End();

    ImGui::Begin("Properties");

    ImGui::End();
}
}    // namespace atcg