#pragma once

namespace atcg
{
namespace GUI
{
ATCG_INLINE SceneHierarchyPanel::SceneHierarchyPanel(const atcg::ref_ptr<Scene>& scene) : _scene(scene) {}

ATCG_INLINE void SceneHierarchyPanel::drawEntityNode(Entity entity) {}

ATCG_INLINE void SceneHierarchyPanel::drawSceneProperties() {}

template<typename... Components>
ATCG_INLINE void SceneHierarchyPanel::drawComponents(Entity entity)
{
}

ATCG_INLINE void SceneHierarchyPanel::selectEntity(Entity entity)
{
    _selected_entity   = entity;
    _focues_components = true;
}

template<typename... CustomComponents>
ATCG_INLINE void SceneHierarchyPanel::renderPanel()
{
}
}    // namespace GUI
}    // namespace atcg