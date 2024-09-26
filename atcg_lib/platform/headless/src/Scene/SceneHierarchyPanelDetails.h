#pragma once

namespace atcg
{

template<typename GUIHandler>
SceneHierarchyPanel<GUIHandler>::SceneHierarchyPanel(const atcg::ref_ptr<Scene>& scene)
    : _scene(scene),
      _gui_handler(atcg::make_ref<GUIHandler>(scene))
{
}

template<typename GUIHandler>
void SceneHierarchyPanel<GUIHandler>::drawEntityNode(Entity entity)
{
}

template<typename GUIHandler>
void SceneHierarchyPanel<GUIHandler>::drawSceneProperties()
{
}

template<typename GUIHandler>
template<typename... Components>
ATCG_INLINE void SceneHierarchyPanel<GUIHandler>::drawComponents(Entity entity)
{
}

template<typename GUIHandler>
ATCG_INLINE void SceneHierarchyPanel<GUIHandler>::selectEntity(Entity entity)
{
    _selected_entity   = entity;
    _focues_components = true;
}

template<typename GUIHandler>
template<typename... CustomComponents>
ATCG_INLINE void SceneHierarchyPanel<GUIHandler>::renderPanel()
{
}
}    // namespace atcg