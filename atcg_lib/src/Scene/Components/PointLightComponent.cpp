#include <Scene/Components/PointLightComponent.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{

void ComponentRenderer<PointLightComponent>::renderComponent(atcg::RendererSystem* _renderer,
                                                             Entity entity,
                                                             const atcg::ref_ptr<Camera>& camera,
                                                             atcg::Dictionary& auxiliary) const
{
    auto& transform   = entity.getComponent<atcg::TransformComponent>();
    auto& point_light = entity.getComponent<atcg::PointLightComponent>();

    const auto& shader = _renderer->getShaderManager()->getShader("circle");
    shader->setInt("entityID", entity.entity_handle());
    _renderer->drawCircle(transform.getPosition(), 0.1f, 1.0f, point_light.color, camera);
}

namespace GUI
{
void ComponentGUIRenderer<PointLightComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               PointLightComponent& _component) const
{
#ifndef ATCG_HEADLESS
    PointLightComponent component = _component;
    bool updated = ImGui::DragFloat("Intensity##PointLight", &component.intensity, 0.01f, 0.0f, FLT_MAX);
    updated      = ImGui::ColorEdit3("Color##PointLight", glm::value_ptr(component.color)) || updated;
    updated      = ImGui::Checkbox("Cast Shadows##PointLight", &component.cast_shadow) || updated;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<PointLightComponent>>(scene, entity);
        _component = component;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(PointLightComponent);
}    // namespace atcg