#include <Scene/SceneHierarchyPanel.h>

#include <imgui.h>
#include <Scene/Components.h>

#include <string.h>

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

void SceneHierarchyPanel::drawComponents(Entity entity)
{
    auto& tag      = entity.getComponent<NameComponent>().name;
    std::string id = std::to_string(entity.getComponent<IDComponent>().ID);
    std::stringstream label;

    char buffer[256];
    memset(buffer, 0, sizeof(buffer));
    // TODO: This is kind of clunky if we keep the box selected
    strncpy_s(buffer, sizeof(buffer), tag.c_str(), sizeof(buffer));
    label << "Name##" << id;
    if(ImGui::InputText(label.str().c_str(), buffer, sizeof(buffer))) { tag = std::string(buffer); }

    if(entity.hasComponent<TransformComponent>())
    {
        auto& transform    = entity.getComponent<TransformComponent>();
        glm::vec3 position = transform.getPosition();
        label.str(std::string());
        label << "Position##" << id;
        if(ImGui::InputFloat3(label.str().c_str(), glm::value_ptr(position))) { transform.setPosition(position); }
        glm::vec3 scale = transform.getScale();
        label.str(std::string());
        label << "Scale##" << id;
        if(ImGui::InputFloat3(label.str().c_str(), glm::value_ptr(scale))) { transform.setScale(scale); }
        glm::vec3 rotation = glm::degrees(transform.getRotation());
        label.str(std::string());
        label << "Rotation##" << id;
        if(ImGui::InputFloat3(label.str().c_str(), glm::value_ptr(rotation)))
        {
            transform.setRotation(glm::radians(rotation));
        }
    }
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

    if(_selected_entity) { drawComponents(_selected_entity); }

    ImGui::End();
}
}    // namespace atcg