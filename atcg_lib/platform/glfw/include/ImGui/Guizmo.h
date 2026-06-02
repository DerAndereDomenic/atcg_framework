#pragma once

#include <Core/API.h>
#include <imgui.h>
#include <ImGuizmo.h>

#include <Core/Memory.h>
#include <Scene/Entity.h>
#include <Renderer/PerspectiveCamera.h>

namespace atcg
{
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
                         ImGuizmo::OPERATION operation,
                         const atcg::ref_ptr<PerspectiveCamera>& camera);
}    // namespace atcg