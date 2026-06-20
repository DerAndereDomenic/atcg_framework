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

/**
 * @brief Draw a guizmo of the selected entity
 *
 * @param scene The scene
 * @param entity The entity
 * @param operation The guizmo operation
 * @param camera The camera to draw from
 */
ATCG_API void drawGuizmo(const atcg::ref_ptr<Scene>& scene,
                         Entity entity,
                         GuizmoOperation operation,
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