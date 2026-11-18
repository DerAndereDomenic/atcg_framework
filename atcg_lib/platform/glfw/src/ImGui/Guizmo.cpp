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

/**
 * @brief Compute the matrix that is manipulated by the guizmo.
 * This is the model matrix of the pivot entity with its translation replaced by the mean position of all entities. For
 * a single entity, this is exactly its model matrix.
 *
 * The matrix is recomputed every frame from the current transforms. This works because ImGuizmo applies translations
 * and rotations relative to the matrix it receives in each frame, and scales relative to the scale of the matrix at the
 * start of the drag. Since the pivot entity is scaled along with the guizmo, its current scale always matches what
 * ImGuizmo expects, and new_matrix * inverse(old_matrix) is the transformation of the current frame.
 *
 * @param entities The entities (all have a TransformComponent)
 * @param pivot_entity The entity that defines the orientation and scale of the guizmo
 *
 * @return The guizmo matrix
 */
ATCG_INLINE glm::mat4 computeGuizmoMatrix(const std::vector<Entity>& entities, Entity pivot_entity)
{
    glm::vec3 center(0);
    for(auto entity: entities)
    {
        center += entity.getComponent<TransformComponent>().getPosition();
    }
    center /= (float)entities.size();

    glm::mat4 matrix = pivot_entity.getComponent<TransformComponent>().getModel();
    matrix[3]        = glm::vec4(center, 1.0f);

    return matrix;
}
}    // namespace detail

void drawGuizmo(const atcg::ref_ptr<Scene>& scene,
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

    // Only entities with a transform can be manipulated. The active entity is the last element of the selection.
    std::vector<Entity> entities;
    for(auto entity: scene->getSelectedEntities())
    {
        if(entity && entity.hasComponent<atcg::TransformComponent>())
        {
            entities.push_back(entity);
        }
    }

    if(!entities.empty())
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

        // The guizmo is placed at the center of the selection and oriented like the active entity
        glm::mat4 before = detail::computeGuizmoMatrix(entities, entities.back());
        glm::mat4 matrix = before;

        bool manipulated = ImGuizmo::Manipulate(glm::value_ptr(camera_view),
                                                glm::value_ptr(camera_projection),
                                                detail::toImGuizmoOperation(operation),
                                                detail::toImGuizmoMode(mode),
                                                glm::value_ptr(matrix));

        bool deactived = ImGui::IsItemDeactivated();

        if(manipulated && !atcg::RevisionStack::isRecording())
        {
            atcg::RevisionStack::startRecording<MultiEntityRevision<ComponentEditedRevision<TransformComponent>>>(
                scene,
                entities);
        }

        if(manipulated)
        {
            if(entities.size() == 1)
            {
                // The guizmo matrix is the model matrix of the entity
                entities.front().getComponent<TransformComponent>().setModel(matrix);
            }
            else
            {
                // Apply the transformation of this frame to every entity
                glm::mat4 delta = matrix * glm::inverse(before);
                for(auto entity: entities)
                {
                    auto& transform = entity.getComponent<TransformComponent>();
                    transform.setModel(delta * transform.getModel());
                }
            }
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
