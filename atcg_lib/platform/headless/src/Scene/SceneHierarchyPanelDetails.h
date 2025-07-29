#pragma once

namespace atcg
{
namespace GUI
{
ATCG_INLINE void SceneHierarchyPanel::drawEntityNode(const atcg::ref_ptr<Scene>& scene, Entity entity) {}

template<typename... Components>
ATCG_INLINE void SceneHierarchyPanel::drawComponents(const atcg::ref_ptr<Scene>& scene, Entity entity)
{
}

ATCG_INLINE void SceneHierarchyPanel::selectEntity(Entity entity)
{
    _selected_entity   = entity;
    _focues_components = true;
}

template<typename... CustomComponents>
ATCG_INLINE void SceneHierarchyPanel::renderPanel(const atcg::ref_ptr<Scene>& scene)
{
}
}    // namespace GUI
}    // namespace atcg