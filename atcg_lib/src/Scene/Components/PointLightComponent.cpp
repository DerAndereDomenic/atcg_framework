#include <Scene/Components/PointLightComponent.h>
#include <Asset/AssetManagerSystem.h>
#include <Scene/ComponentRegistry.h>
#include <Scene/Components/TransformComponent.h>

#define POINT_LIGHT_KEY     "PointLight"
#define INTENSITY_KEY       "Intensity"
#define CAST_SHADOWS_KEY    "CastShadow"
#define RECEIVE_SHADOWS_KEY "ReceiveShadow"
#define COLOR_KEY           "Color"

namespace atcg
{

void ComponentRenderer<PointLightComponent>::renderComponent(atcg::RendererSystem* _renderer,
                                                             Entity entity,
                                                             const atcg::ref_ptr<Camera>& camera,
                                                             atcg::Dictionary& auxiliary) const
{
    auto& transform   = entity.getComponent<atcg::TransformComponent>();
    auto& point_light = entity.getComponent<atcg::PointLightComponent>();

    const float radius    = 0.1f;
    const float thickness = 1.0f;

    auto override_shader = auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader", nullptr);
    if(override_shader)
    {
        // Draw the camera-facing billboard ourselves so that the override shader is used. The shader is expected to
        // use the vertex stage of circle.vs (uniforms: position, radius, V, P).
        auto quad = AssetManager::getQuadMesh();

        override_shader->setVec3("position", transform.getPosition());
        override_shader->setFloat("radius", radius);
        override_shader->setFloat("thickness", thickness);
        override_shader->setVec3("flat_color", point_light.color);
        override_shader->setInt("entityID", entity.entity_handle());

        GraphicsPipeline pipeline = GraphicsPipeline()
                                        .setShader(override_shader)
                                        .setPrimitiveTopology(PrimitiveTopology::ATCG_TRIANGLES)
                                        .setRasterizerState(RasterizerState().enableCulling(false));

        _renderer->drawVAO(quad->getVerticesArray(), camera, glm::mat4(1), pipeline, quad->n_vertices());
        return;
    }

    const auto& shader = _renderer->getShaderManager()->getShader("circle");
    shader->setInt("entityID", entity.entity_handle());
    _renderer
        ->drawCircle(transform.getPosition(), radius, thickness, point_light.color, camera, entity.entity_handle());
}

namespace Serialization
{
void ComponentSerializer<PointLightComponent>::serialize_component(const std::string& file_path,
                                                                   const atcg::ref_ptr<Scene>& scene,
                                                                   Entity entity,
                                                                   PointLightComponent& component,
                                                                   nlohmann::json& j) const
{
    j[POINT_LIGHT_KEY][INTENSITY_KEY] = component.intensity;
    j[POINT_LIGHT_KEY][COLOR_KEY] = nlohmann::json::array({component.color.x, component.color.y, component.color.z});
    j[POINT_LIGHT_KEY][CAST_SHADOWS_KEY] = component.cast_shadow;
}


void ComponentSerializer<PointLightComponent>::deserialize_component(const std::string& file_path,
                                                                     const atcg::ref_ptr<Scene>& scene,
                                                                     Entity entity,
                                                                     nlohmann::json& j) const
{
    if(!j.contains(POINT_LIGHT_KEY))
    {
        return;
    }

    auto& point_light           = j[POINT_LIGHT_KEY];
    auto& renderComponent       = entity.addComponent<PointLightComponent>();
    renderComponent.intensity   = point_light.value(INTENSITY_KEY, 1.0f);
    auto color                  = point_light.value(COLOR_KEY, std::vector<float> {1.0f, 1.0f, 1.0f});
    renderComponent.color       = glm::make_vec3(color.data());
    renderComponent.cast_shadow = point_light.value(CAST_SHADOWS_KEY, true);
}

}    // namespace Serialization


namespace GUI
{
void ComponentGUIRenderer<PointLightComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               PointLightComponent& _component) const
{
#ifndef ATCG_HEADLESS
    bool deactivated              = false;
    PointLightComponent component = _component;
    bool updated = ImGui::DragFloat("Intensity##PointLight", &component.intensity, 0.01f, 0.0f, FLT_MAX);
    deactivated  = ImGui::IsItemDeactivated() || deactivated;
    updated      = ImGui::ColorEdit3("Color##PointLight", glm::value_ptr(component.color)) || updated;
    deactivated  = ImGui::IsItemDeactivated() || deactivated;
    updated      = ImGui::Checkbox("Cast Shadows##PointLight", &component.cast_shadow) || updated;
    deactivated  = ImGui::IsItemDeactivated() || deactivated;

    if(updated && !atcg::RevisionStack::isRecording())
    {
        RevisionStack::startRecording<ComponentEditedRevision<PointLightComponent>>(scene, entity);
    }

    if(updated)
    {
        _component = component;
    }

    if(deactivated && atcg::RevisionStack::isRecording())
    {
        RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

void PointLightComponent::registerComponent(ComponentRegistry::Registry* registry)
{
    ATCG_REGISTER_COMPONENT(registry, "PointLight", PointLightComponent);
}

}    // namespace atcg