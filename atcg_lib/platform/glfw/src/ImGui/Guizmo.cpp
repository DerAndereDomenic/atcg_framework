#include <ImGui/Guizmo.h>

#include <Core/Application.h>
#include <Scene/Components.h>
#include <Scene/RevisionStack.h>

#include <ImGuizmo.h>

namespace atcg
{

namespace detail
{
ATCG_INLINE ImGuizmo::OPERATION toImGuizmoOperation(GuizmoOperation operation)
{
    switch(operation)
    {
        case TRANSLATE:
            return ImGuizmo::TRANSLATE;
        case ROTATE:
            return ImGuizmo::ROTATE;
        case SCALE:
            return ImGuizmo::SCALE;
        default:
            return ImGuizmo::TRANSLATE;
    }
}

ATCG_INLINE ImGuizmo::MODE toImGuizmoMode(GuizmoMode mode)
{
    switch(mode)
    {
        case LOCAL:
            return ImGuizmo::LOCAL;
        case WORLD:
            return ImGuizmo::WORLD;
        default:
            return ImGuizmo::LOCAL;
    }
}

}    // namespace detail

void drawGuizmo(const atcg::ref_ptr<Scene>& scene,
                Entity entity,
                GuizmoOperation operation,
                GuizmoMode mode,
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
                                                detail::toImGuizmoOperation(operation),
                                                detail::toImGuizmoMode(mode),
                                                glm::value_ptr(model));

        bool deactived = ImGui::IsItemDeactivated();

        if(manipulated && !atcg::RevisionStack::isRecording())
        {
            atcg::RevisionStack::startRecording<ComponentEditedRevision<TransformComponent>>(scene, entity);
        }

        if(manipulated)
        {
            transform.setModel(model);
        }

        if(deactived && atcg::RevisionStack::isRecording())
        {
            atcg::RevisionStack::endRecording();
        }
    }

    if(useViewports)
    {
        ImGui::End();
        ImGui::PopStyleVar();
    }
}

bool isUsingGuizmo()
{
    return ImGuizmo::IsUsing();
}

bool isOverGuizmo()
{
    return ImGuizmo::IsOver();
}

}    // namespace atcg