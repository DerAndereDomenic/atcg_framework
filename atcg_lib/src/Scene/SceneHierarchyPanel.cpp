#include <Scene/SceneHierarchyPanel.h>

#include <imgui.h>
#include <Scene/Components.h>
#include <Scene/Entity.h>

namespace atcg
{

namespace detail
{
void drawEntityNode(Entity entity)
{
    auto& tag = entity.getComponent<NameComponent>().name;

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Bullet;
    flags |= ImGuiTreeNodeFlags_SpanAvailWidth;
    bool opened = ImGui::TreeNodeEx((void*)(uint64_t)(uint32_t)entity, flags, tag.c_str());
    // if(ImGui::IsItemClicked()) { m_SelectionContext = entity; }

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
}    // namespace detail


SceneHierarchyPanel::SceneHierarchyPanel(const atcg::ref_ptr<Scene>& scene) : _scene(scene) {}

void SceneHierarchyPanel::renderPanel() const
{
    ImGui::Begin("Scene Hierarchy");

    for(auto e: _scene->getAllEntitiesWith<NameComponent>())
    {
        Entity entity(e, _scene.get());
        detail::drawEntityNode(entity);
    }

    ImGui::End();

    ImGui::Begin("Properties");

    ImGui::End();
}
}    // namespace atcg