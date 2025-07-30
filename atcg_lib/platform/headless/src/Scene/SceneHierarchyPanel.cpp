#include <Scene/SceneHierarchyPanel.h>

namespace atcg
{
namespace GUI
{
void SceneHierarchyPanel::drawEntityNode(const atcg::ref_ptr<Scene>& scene, Entity entity) {}

void SceneHierarchyPanel::drawComponents(const atcg::ref_ptr<Scene>& scene, Entity entity) {}

void SceneHierarchyPanel::selectEntity(Entity entity)
{
    _selected_entity   = entity;
    _focues_components = true;
}

void SceneHierarchyPanel::renderPanel(const atcg::ref_ptr<Scene>& scene) {}
}    // namespace GUI
}    // namespace atcg