#pragma once

#include <imgui.h>
#include <ImGuizmo.h>
#include <Scene/Entity.h>
#include <Renderer/PerspectiveCamera.h>

namespace atcg
{
/**
 * @brief Draw a guizmo of the selected entity
 *
 * @param entity The entity
 * @param operation The guizmo operation
 * @param camera The camera to draw from
 */
void drawGuizmo(Entity entity, ImGuizmo::OPERATION operation, const atcg::ref_ptr<PerspectiveCamera>& camera);
}    // namespace atcg