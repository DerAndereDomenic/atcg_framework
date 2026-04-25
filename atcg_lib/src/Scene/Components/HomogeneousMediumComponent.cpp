#include <Scene/Components/HomogeneousMediumComponent.h>
#include <Scene/ComponentRegistry.h>

#define HOMOGENEOUS_MEDIUM_KEY "Homogeneous Medium"
#define ALBEDO_KEY             "albedo"
#define DENSITY_KEY            "density"
#define G_KEY                  "g"
#define LE_KEY                 "Le"
#define LE_COLOR_KEY           "Le_color"

namespace atcg
{

namespace Serialization
{
void ComponentSerializer<HomogeneousMediumComponent>::serialize_component(const std::string& file_path,
                                                                          const atcg::ref_ptr<Scene>& scene,
                                                                          Entity entity,
                                                                          HomogeneousMediumComponent& component,
                                                                          nlohmann::json& j) const
{
    glm::vec3 albedo   = component.albedo;
    float density      = component.density;
    float g            = component.g;
    float Le           = component.Le;
    glm::vec3 Le_color = component.Le_color;

    j[HOMOGENEOUS_MEDIUM_KEY][ALBEDO_KEY]   = nlohmann::json::array({albedo.x, albedo.y, albedo.z});
    j[HOMOGENEOUS_MEDIUM_KEY][DENSITY_KEY]  = density;
    j[HOMOGENEOUS_MEDIUM_KEY][G_KEY]        = g;
    j[HOMOGENEOUS_MEDIUM_KEY][LE_KEY]       = Le;
    j[HOMOGENEOUS_MEDIUM_KEY][LE_COLOR_KEY] = nlohmann::json::array({Le_color.x, Le_color.y, Le_color.z});
}


void ComponentSerializer<HomogeneousMediumComponent>::deserialize_component(const std::string& file_path,
                                                                            const atcg::ref_ptr<Scene>& scene,
                                                                            Entity entity,
                                                                            nlohmann::json& j) const
{
    if(!j.contains(HOMOGENEOUS_MEDIUM_KEY))
    {
        return;
    }

    std::vector<float> albedo   = j[HOMOGENEOUS_MEDIUM_KEY].value(ALBEDO_KEY, std::vector<float> {1.0f, 1.0f, 1.0f});
    std::vector<float> Le_color = j[HOMOGENEOUS_MEDIUM_KEY].value(LE_COLOR_KEY, std::vector<float> {1.0f, 1.0f, 1.0f});

    auto& component    = entity.addComponent<HomogeneousMediumComponent>();
    component.albedo   = glm::make_vec3(albedo.data());
    component.g        = j[HOMOGENEOUS_MEDIUM_KEY][G_KEY];
    component.Le       = j[HOMOGENEOUS_MEDIUM_KEY][LE_KEY];
    component.Le_color = glm::make_vec3(Le_color.data());
    component.density  = j[HOMOGENEOUS_MEDIUM_KEY][DENSITY_KEY];
}
}    // namespace Serialization

namespace GUI
{
void ComponentGUIRenderer<HomogeneousMediumComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                      Entity entity,
                                                                      HomogeneousMediumComponent& component) const
{
#ifndef ATCG_HEADLESS
    HomogeneousMediumComponent _component = component;

    bool updated = false;
    updated      = ImGui::DragFloat("Density##homogen", &_component.density, 0.05f, 0.0f, 50.0f) || updated;
    updated      = ImGui::DragFloat("g##homogen", &_component.g, 0.01f, -1.0f, 1.0f) || updated;
    updated      = ImGui::ColorEdit3("albedo##homogen", glm::value_ptr(_component.albedo)) || updated;
    updated      = ImGui::DragFloat("Le##homogen", &_component.Le, 0.01f, 0.0f, 100.0f) || updated;
    updated      = ImGui::ColorEdit3("LeColor##homogen", glm::value_ptr(_component.Le_color)) || updated;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<HomogeneousMediumComponent>>(scene, entity);
        component = _component;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT_DRAW(HomogeneousMediumComponent);
ATCG_REGISTER_COMPONENT_STORE(HomogeneousMediumComponent);
ATCG_REGISTER_COMPONENT_SERIALIZATION(HomogeneousMediumComponent);
}    // namespace atcg