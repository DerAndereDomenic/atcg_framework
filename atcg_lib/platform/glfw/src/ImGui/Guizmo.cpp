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

    if(entity && (entity.hasComponent<atcg::TransformComponent>()))
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

        atcg::TransformComponent& transform = entity.getComponent<atcg::TransformComponent>();
        model                               = transform.getModel();

        bool manipulated = ImGuizmo::Manipulate(glm::value_ptr(camera_view),
                                                glm::value_ptr(camera_projection),
                                                operation,
                                                ImGuizmo::LOCAL,
                                                glm::value_ptr(model));

        if(manipulated)
        {
            atcg::RevisionStack::startRecording<ComponentEditedRevision<TransformComponent>>(scene, entity);
            atcg::TransformComponent& transform = entity.getComponent<atcg::TransformComponent>();
            transform.setModel(model);
            atcg::RevisionStack::endRecording();
        }
    }

    if(useViewports)
    {
        ImGui::End();
        ImGui::PopStyleVar();
    }
}
}    // namespace atcg