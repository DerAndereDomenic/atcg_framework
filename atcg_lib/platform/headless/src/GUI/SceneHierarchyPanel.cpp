#include <GUI/SceneHierarchyPanel.h>

#include <Scene/Scene.h>

namespace atcg
{
namespace GUI
{
void SceneHierarchyPanel::drawEntityNode(const atcg::ref_ptr<Scene>& scene, Entity entity) {}

void SceneHierarchyPanel::drawComponents(const atcg::ref_ptr<Scene>& scene, Entity entity) {}

void SceneHierarchyPanel::selectEntity(Entity entity)
{
    _focues_components = true;

    if(entity && entity.scene())
    {
        entity.scene()->selectEntity(entity);
    }
}

void SceneHierarchyPanel::renderPanel(const atcg::ref_ptr<Scene>& scene) {}
}    // namespace GUI
}    // namespace atcg