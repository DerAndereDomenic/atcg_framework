#pragma once

#include <Core/API.h>
#include <imgui.h>

#include <Core/Memory.h>
#include <Scene/Entity.h>
#include <Renderer/PerspectiveCamera.h>

namespace atcg
{

enum GuizmoOperation
{
    TRANSLATE,
    ROTATE,
    SCALE
};

enum GuizmoMode
{
    LOCAL,
    WORLD
};

/**
 * @brief Draw a guizmo for all selected entities of the scene.
 * If one entity is selected, the guizmo manipulates its transform. If multiple entities are selected, the guizmo is
 * placed at the mean position of the selection and transforms all selected entities around it. In local mode, the
 * guizmo uses the orientation of the active entity.
 *
 * @param scene The scene
 * @param operation The guizmo operation
 * @param mode The guizmo mode
 * @param camera The camera to draw from
 */
ATCG_API void drawGuizmo(const atcg::ref_ptr<Scene>& scene,
                         GuizmoOperation operation,
                         GuizmoMode mode,
                         const atcg::ref_ptr<PerspectiveCamera>& camera);

/**
 * @brief Check if the guizmo is being used
 *
 * @return true if the guizmo is being used, false otherwise
 */
ATCG_API bool isUsingGuizmo();

/**
 * @brief Check if mouse if over guizmo
 *
 * @return true if the mouse is over the guizmo, false otherwise
 */
ATCG_API bool isOverGuizmo();
}    // namespace atcg