#include <Scene/Components/HeterogeneousMediumComponent.h>
#include <Scene/ComponentRegistry.h>

#define HETEROGENEOUS_MEDIUM_KEY "Heterogeneous Medium"
#define DENSITY_GRID_KEY         "density_grid"
#define ALBEDO_GRID_KEY          "albedo_grid"
#define EMISSION_GRID_KEY        "emission_grid"
#define G_KEY                    "g"
#define GRID_KEY                 "grid"
#define SCALE_KEY                "scale"
#define BBOX_KEY                 "bbox"
#define BBOX_MIN_KEY             "min"
#define BBOX_MAX_KEY             "max"

namespace atcg
{

namespace Serialization
{
void ComponentSerializer<HeterogeneousMediumComponent>::serialize_component(const std::string& file_path,
                                                                            const atcg::ref_ptr<Scene>& scene,
                                                                            Entity entity,
                                                                            HeterogeneousMediumComponent& component,
                                                                            nlohmann::json& j) const
{
    nlohmann::json j_density_grid;
    nlohmann::json j_albedo_grid;
    nlohmann::json j_emission_grid;

    j_density_grid[GRID_KEY]  = (uint64_t)component.density_grid.handle;
    j_density_grid[SCALE_KEY] = component.density_grid.scale;
    nlohmann::json j_bbox_density;
    j_bbox_density[BBOX_MIN_KEY] = nlohmann::json::array(
        {component.density_grid.bbox.min.x, component.density_grid.bbox.min.y, component.density_grid.bbox.min.z});
    j_bbox_density[BBOX_MAX_KEY] = nlohmann::json::array(
        {component.density_grid.bbox.max.x, component.density_grid.bbox.max.y, component.density_grid.bbox.max.z});
    j_density_grid[BBOX_KEY] = j_bbox_density;

    j_albedo_grid[GRID_KEY]  = (uint64_t)component.albedo_grid.handle;
    j_albedo_grid[SCALE_KEY] = component.albedo_grid.scale;
    nlohmann::json j_bbox_albedo;
    j_bbox_albedo[BBOX_MIN_KEY] = nlohmann::json::array(
        {component.albedo_grid.bbox.min.x, component.albedo_grid.bbox.min.y, component.albedo_grid.bbox.min.z});
    j_bbox_albedo[BBOX_MAX_KEY] = nlohmann::json::array(
        {component.albedo_grid.bbox.max.x, component.albedo_grid.bbox.max.y, component.albedo_grid.bbox.max.z});
    j_albedo_grid[BBOX_KEY] = j_bbox_albedo;

    j_emission_grid[GRID_KEY]  = (uint64_t)component.emission_grid.handle;
    j_emission_grid[SCALE_KEY] = component.emission_grid.scale;
    nlohmann::json j_bbox_emission;
    j_bbox_emission[BBOX_MIN_KEY] = nlohmann::json::array(
        {component.emission_grid.bbox.min.x, component.emission_grid.bbox.min.y, component.emission_grid.bbox.min.z});
    j_bbox_emission[BBOX_MAX_KEY] = nlohmann::json::array(
        {component.emission_grid.bbox.max.x, component.emission_grid.bbox.max.y, component.emission_grid.bbox.max.z});
    j_emission_grid[BBOX_KEY] = j_bbox_emission;

    j[HETEROGENEOUS_MEDIUM_KEY][DENSITY_GRID_KEY]  = j_density_grid;
    j[HETEROGENEOUS_MEDIUM_KEY][ALBEDO_GRID_KEY]   = j_albedo_grid;
    j[HETEROGENEOUS_MEDIUM_KEY][EMISSION_GRID_KEY] = j_emission_grid;
    j[HETEROGENEOUS_MEDIUM_KEY][G_KEY]             = component.g;
}


void ComponentSerializer<HeterogeneousMediumComponent>::deserialize_component(const std::string& file_path,
                                                                              const atcg::ref_ptr<Scene>& scene,
                                                                              Entity entity,
                                                                              nlohmann::json& j) const
{
    if(!j.contains(HETEROGENEOUS_MEDIUM_KEY))
    {
        return;
    }

    auto& component = entity.addComponent<HeterogeneousMediumComponent>();

    // --- Density grid ---
    const auto& j_density         = j[HETEROGENEOUS_MEDIUM_KEY][DENSITY_GRID_KEY];
    component.density_grid.handle = (AssetHandle)j_density[GRID_KEY].get<uint64_t>();
    component.density_grid.scale  = j_density[SCALE_KEY].get<float>();

    const auto& j_bbox_density      = j_density[BBOX_KEY];
    component.density_grid.bbox.min = glm::vec3(j_bbox_density[BBOX_MIN_KEY][0].get<float>(),
                                                j_bbox_density[BBOX_MIN_KEY][1].get<float>(),
                                                j_bbox_density[BBOX_MIN_KEY][2].get<float>());
    component.density_grid.bbox.max = glm::vec3(j_bbox_density[BBOX_MAX_KEY][0].get<float>(),
                                                j_bbox_density[BBOX_MAX_KEY][1].get<float>(),
                                                j_bbox_density[BBOX_MAX_KEY][2].get<float>());

    // --- Albedo grid ---
    const auto& j_albedo         = j[HETEROGENEOUS_MEDIUM_KEY][ALBEDO_GRID_KEY];
    component.albedo_grid.handle = (AssetHandle)j_albedo[GRID_KEY].get<uint64_t>();
    component.albedo_grid.scale  = j_albedo[SCALE_KEY].get<float>();

    const auto& j_bbox_albedo      = j_albedo[BBOX_KEY];
    component.albedo_grid.bbox.min = glm::vec3(j_bbox_albedo[BBOX_MIN_KEY][0].get<float>(),
                                               j_bbox_albedo[BBOX_MIN_KEY][1].get<float>(),
                                               j_bbox_albedo[BBOX_MIN_KEY][2].get<float>());
    component.albedo_grid.bbox.max = glm::vec3(j_bbox_albedo[BBOX_MAX_KEY][0].get<float>(),
                                               j_bbox_albedo[BBOX_MAX_KEY][1].get<float>(),
                                               j_bbox_albedo[BBOX_MAX_KEY][2].get<float>());

    // --- Emission grid ---
    const auto& j_emission         = j[HETEROGENEOUS_MEDIUM_KEY][EMISSION_GRID_KEY];
    component.emission_grid.handle = (AssetHandle)j_emission[GRID_KEY].get<uint64_t>();
    component.emission_grid.scale  = j_emission[SCALE_KEY].get<float>();

    const auto& j_bbox_emission      = j_emission[BBOX_KEY];
    component.emission_grid.bbox.min = glm::vec3(j_bbox_emission[BBOX_MIN_KEY][0].get<float>(),
                                                 j_bbox_emission[BBOX_MIN_KEY][1].get<float>(),
                                                 j_bbox_emission[BBOX_MIN_KEY][2].get<float>());
    component.emission_grid.bbox.max = glm::vec3(j_bbox_emission[BBOX_MAX_KEY][0].get<float>(),
                                                 j_bbox_emission[BBOX_MAX_KEY][1].get<float>(),
                                                 j_bbox_emission[BBOX_MAX_KEY][2].get<float>());

    component.g = j[HETEROGENEOUS_MEDIUM_KEY][G_KEY].get<float>();
}
}    // namespace Serialization

namespace GUI
{
void ComponentGUIRenderer<HeterogeneousMediumComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                        Entity entity,
                                                                        HeterogeneousMediumComponent& component) const
{
#ifndef ATCG_HEADLESS
    HeterogeneousMediumComponent _component = component;

    bool updated = false;

    ImGui::Text("Density");
    auto new_handle                = displayTexture3DSelection("densitytexture3d", _component.density_grid.handle);
    updated                        = updated || (new_handle != _component.density_grid.handle);
    _component.density_grid.handle = new_handle;
    updated =
        ImGui::DragFloat("Desity Scale##texture3d", &_component.density_grid.scale, 0.01f, 0.0f, 10.0f) || updated;
    ImGui::Text("Bounding Box");
    updated = ImGui::DragFloat3("Min##density", glm::value_ptr(_component.density_grid.bbox.min), 0.05f) || updated;
    updated = ImGui::DragFloat3("Max##density", glm::value_ptr(_component.density_grid.bbox.max), 0.05f) || updated;

    ImGui::Separator();
    ImGui::Text("Albedo");
    new_handle                    = displayTexture3DSelection("albedotexture3d", _component.albedo_grid.handle);
    updated                       = updated || (new_handle != _component.albedo_grid.handle);
    _component.albedo_grid.handle = new_handle;
    updated = ImGui::DragFloat("Albedo Scale##texture3d", &_component.albedo_grid.scale, 0.01f, 0.0f, 1.0f) || updated;
    ImGui::Text("Bounding Box");
    updated = ImGui::DragFloat3("Min##albedo", glm::value_ptr(_component.albedo_grid.bbox.min), 0.05f) || updated;
    updated = ImGui::DragFloat3("Max##albedo", glm::value_ptr(_component.albedo_grid.bbox.max), 0.05f) || updated;

    ImGui::Separator();
    ImGui::Text("Emission");
    new_handle                      = displayTexture3DSelection("emissiontexture3d", _component.emission_grid.handle);
    updated                         = updated || (new_handle != _component.emission_grid.handle);
    _component.emission_grid.handle = new_handle;
    updated =
        ImGui::DragFloat("Emission Scale##texture3d", &_component.emission_grid.scale, 0.01f, 0.0f, 10.0f) || updated;
    ImGui::Text("Bounding Box");
    updated = ImGui::DragFloat3("Min##emission", glm::value_ptr(_component.emission_grid.bbox.min), 0.05f) || updated;
    updated = ImGui::DragFloat3("Max##emission", glm::value_ptr(_component.emission_grid.bbox.max), 0.05f) || updated;
    updated = ImGui::DragFloat("g##het", &_component.g, 0.01f, -1.0f, 1.0f) || updated;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<HeterogeneousMediumComponent>>(scene, entity);
        component = _component;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT_DRAW(HeterogeneousMediumComponent);
ATCG_REGISTER_COMPONENT_STORE(HeterogeneousMediumComponent);
ATCG_REGISTER_COMPONENT_SERIALIZATION(HeterogeneousMediumComponent);
}    // namespace atcg