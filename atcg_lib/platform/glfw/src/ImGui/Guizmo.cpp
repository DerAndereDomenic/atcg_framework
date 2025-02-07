#include <ImGui/Guizmo.h>

#include <Core/Application.h>
#include <Scene/Components.h>
#include <Scene/RevisionStack.h>

namespace atcg
{
void drawGuizmo(const atcg::ref_ptr<Scene>& scene,
                Entity entity,
                ImGuizmo::OPERATION operation,
                const atcg::ref_ptr<PerspectiveCamera>& camera)
{
    bool useViewports        = atcg::Application::get()->getImGuiLayer()->dockspaceEnabled();
    const auto& window       = atcg::Application::get()->getWindow();
    glm::ivec2 window_pos    = window->getPosition();
    glm::ivec2 viewport_pos  = atcg::Application::get()->getViewportPosition();
    glm::ivec2 viewport_size = atcg::Application::get()->getViewportSize();

    if(useViewports)
    {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2 {0, 0});
        ImGui::Begin("Viewport");
    }
    if(entity && (entity.hasAnyComponent<atcg::TransformComponent, atcg::CameraComponent>()))
    {
        ImGuizmo::SetOrthographic(false);
        ImGuizmo::BeginFrame();
        if(useViewports)
        {
            ImGuizmo::SetDrawlist();
        }

        ImGuizmo::SetRect(window_pos.x + viewport_pos.x,
                          window_pos.y + viewport_pos.y,
                          viewport_size.x,
                          viewport_size.y);

        glm::mat4 camera_projection = camera->getProjection();
        glm::mat4 camera_view       = camera->getView();

        glm::mat4 model;
        bool has_transform = entity.hasComponent<atcg::TransformComponent>();
        if(has_transform)
        {
            atcg::TransformComponent& transform = entity.getComponent<atcg::TransformComponent>();
            model                               = transform.getModel();
        }
        else
        {
            atcg::CameraComponent& cam_component = entity.getComponent<atcg::CameraComponent>();
            auto cam = std::dynamic_pointer_cast<atcg::PerspectiveCamera>(cam_component.camera);
            model    = cam->getAsTransform();
        }

        float scale_x = 1.0f, scale_y = 1.0f, scale_z = 1.0f;
        if(operation != ImGuizmo::SCALE)
        {
            scale_x = glm::length(glm::vec3(model[0]));
            scale_y = glm::length(glm::vec3(model[1]));
            scale_z = glm::length(glm::vec3(model[2]));

            model = model * glm::scale(glm::vec3(1.0f / scale_x, 1.0f / scale_y, 1.0f / scale_z));
        }

        bool manipulated = ImGuizmo::Manipulate(glm::value_ptr(camera_view),
                                                glm::value_ptr(camera_projection),
                                                operation,
                                                ImGuizmo::LOCAL,
                                                glm::value_ptr(model));

        if(manipulated)
        {
            model = model * glm::scale(glm::vec3(scale_x, scale_y, scale_z));

            if(has_transform)
            {
                auto recorder =
                    atcg::RevisionStack::recordRevision<ComponentEditedRevision<TransformComponent>>(scene, entity);
                atcg::TransformComponent& transform = entity.getComponent<atcg::TransformComponent>();
                transform.setModel(model);
            }
            else
            {
                auto recorder =
                    atcg::RevisionStack::recordRevision<ComponentEditedRevision<CameraComponent>>(scene, entity);
                atcg::CameraComponent& cam_component = entity.getComponent<atcg::CameraComponent>();
                auto cam = std::dynamic_pointer_cast<atcg::PerspectiveCamera>(cam_component.camera);
                cam->setFromTransform(model);
            }
        }
    }

    if(useViewports)
    {
        ImGui::End();
        ImGui::PopStyleVar();
    }
}
}    // namespace atcg