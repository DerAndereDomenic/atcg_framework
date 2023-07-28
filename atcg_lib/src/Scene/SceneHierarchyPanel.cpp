#include <Scene/SceneHierarchyPanel.h>

#include <imgui.h>
#include <Scene/Components.h>

#include <string.h>

namespace atcg
{

namespace detail
{
template<typename T, typename UIFunction>
void drawComponent(const std::string& name, Entity entity, UIFunction uiFunction)
{
    const ImGuiTreeNodeFlags treeNodeFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed |
                                             ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap |
                                             ImGuiTreeNodeFlags_FramePadding;

    if(entity.hasComponent<T>())
    {
        auto& component = entity.getComponent<T>();
        bool open       = ImGui::TreeNodeEx((void*)typeid(T).hash_code(), treeNodeFlags, name.c_str());

        if(open)
        {
            uiFunction(component);
            ImGui::TreePop();
        }
    }
}
}    // namespace detail

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
    std::string id = std::to_string(entity.getComponent<IDComponent>().ID);
    std::stringstream label;

    detail::drawComponent<NameComponent>("Name",
                                         entity,
                                         [&](NameComponent& component)
                                         {
                                             std::string tag = component.name;
                                             char buffer[256];
                                             memset(buffer, 0, sizeof(buffer));
                                             strncpy_s(buffer, sizeof(buffer), tag.c_str(), sizeof(buffer));
                                             label << "Name##" << id;
                                             if(ImGui::InputText(label.str().c_str(), buffer, sizeof(buffer)))
                                             {
                                                 tag = std::string(buffer);
                                             }
                                         });

    detail::drawComponent<TransformComponent>("Transform",
                                              entity,
                                              [&](TransformComponent& transform)
                                              {
                                                  glm::vec3 position = transform.getPosition();
                                                  label.str(std::string());
                                                  label << "Position##" << id;
                                                  if(ImGui::InputFloat3(label.str().c_str(), glm::value_ptr(position)))
                                                  {
                                                      transform.setPosition(position);
                                                  }
                                                  glm::vec3 scale = transform.getScale();
                                                  label.str(std::string());
                                                  label << "Scale##" << id;
                                                  if(ImGui::InputFloat3(label.str().c_str(), glm::value_ptr(scale)))
                                                  {
                                                      transform.setScale(scale);
                                                  }
                                                  glm::vec3 rotation = glm::degrees(transform.getRotation());
                                                  label.str(std::string());
                                                  label << "Rotation##" << id;
                                                  if(ImGui::InputFloat3(label.str().c_str(), glm::value_ptr(rotation)))
                                                  {
                                                      transform.setRotation(glm::radians(rotation));
                                                  }
                                              });
    detail::drawComponent<MeshRenderComponent>("Mesh Renderer",
                                               entity,
                                               [&](MeshRenderComponent& component)
                                               {
                                                   glm::vec3 color = component.color;
                                                   label.str(std::string());
                                                   label << "Base Color##mesh" << id;
                                                   if(ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(color)))
                                                   {
                                                       component.color = color;
                                                   }
                                               });
    detail::drawComponent<PointRenderComponent>("Point Renderer",
                                                entity,
                                                [&](PointRenderComponent& component)
                                                {
                                                    glm::vec3 color = component.color;
                                                    label.str(std::string());
                                                    label << "Base Color##point" << id;
                                                    if(ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(color)))
                                                    {
                                                        component.color = color;
                                                    }

                                                    float point_size = component.point_size;
                                                    label.str(std::string());
                                                    label << "Point Size##point" << id;
                                                    if(ImGui::InputFloat(label.str().c_str(), &point_size))
                                                    {
                                                        component.point_size = point_size;
                                                    }
                                                });
    detail::drawComponent<PointSphereRenderComponent>(
        "Point Sphere Renderer",
        entity,
        [&](PointSphereRenderComponent& component)
        {
            glm::vec3 color = component.color;
            label.str(std::string());
            label << "Base Color##pointsphere" << id;
            if(ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(color))) { component.color = color; }

            float point_size = component.point_size;
            label.str(std::string());
            label << "Point Size##pointsphere" << id;
            if(ImGui::InputFloat(label.str().c_str(), &point_size)) { component.point_size = point_size; }
        });

    detail::drawComponent<EdgeRenderComponent>("Edge Renderer",
                                               entity,
                                               [&](EdgeRenderComponent& component)
                                               {
                                                   glm::vec3 color = component.color;
                                                   label.str(std::string());
                                                   label << "Base Color##edge" << id;
                                                   if(ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(color)))
                                                   {
                                                       component.color = color;
                                                   }
                                               });

    detail::drawComponent<EdgeCylinderRenderComponent>(
        "Edge Cylinder Renderer",
        entity,
        [&](EdgeCylinderRenderComponent& component)
        {
            glm::vec3 color = component.color;
            label.str(std::string());
            label << "Base Color##edgecylinder" << id;
            if(ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(color))) { component.color = color; }
        });
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