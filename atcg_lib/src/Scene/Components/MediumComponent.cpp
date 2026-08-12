#include <Scene/Components/MediumComponent.h>
#include <Scene/ComponentRegistry.h>
#include <Utils/Utils.h>

#include <Material/MediumRegistry.h>

#define MEDIUM_KEY "Medium"

#define HOMOGENEOUS_MEDIUM_KEY "Homogeneous Medium"
#define ALBEDO_KEY             "albedo"
#define DENSITY_KEY            "density"
#define G_KEY                  "g"
#define LE_KEY                 "Le"
#define LE_COLOR_KEY           "Le_color"

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

void ComponentRenderer<MediumComponent>::renderComponent(atcg::RendererSystem* _renderer,
                                                         Entity entity,
                                                         const atcg::ref_ptr<Camera>& camera,
                                                         atcg::Dictionary& auxiliary) const
{
    // TODO
}

namespace Serialization
{
void ComponentSerializer<MediumComponent>::serialize_component(const std::string& file_path,
                                                               const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               MediumComponent& component,
                                                               nlohmann::json& j) const
{
    j[MEDIUM_KEY] = (uint64_t)component.medium_handle;
}

void ComponentSerializer<MediumComponent>::deserialize_component(const std::string& file_path,
                                                                 const atcg::ref_ptr<Scene>& scene,
                                                                 Entity entity,
                                                                 nlohmann::json& j) const
{
    if(j.contains(MEDIUM_KEY))
    {
        auto& medium         = entity.addComponent<MediumComponent>();
        medium.medium_handle = (AssetHandle)j[MEDIUM_KEY];
        return;
    }

    if(j.contains(HOMOGENEOUS_MEDIUM_KEY) && j.contains(HETEROGENEOUS_MEDIUM_KEY))
    {
        ATCG_WARN("Entity contains both Homogeneous and Heterogeneous Medium components. Only one should be present. "
                  "-- Skipping deserialization of MediumComponent.");
        return;
    }

    // These two are for backwards compatibility with the HomogeneousMediumComponent
    if(j.contains(HOMOGENEOUS_MEDIUM_KEY))
    {
        auto medium_json = j[HOMOGENEOUS_MEDIUM_KEY];

        auto& component = entity.addComponent<MediumComponent>();

        auto medium = atcg::MediumRegistry::deserializeMedium("Homogeneous", file_path, medium_json);
        atcg::AssetManager::registerAsset(medium, "medium");
        component.medium_handle = medium->handle;
        return;
    }

    if(j.contains(HETEROGENEOUS_MEDIUM_KEY))
    {
        auto medium_json = j[HETEROGENEOUS_MEDIUM_KEY];

        auto& component = entity.addComponent<MediumComponent>();

        auto medium = atcg::MediumRegistry::deserializeMedium("Heterogeneous", file_path, medium_json);
        atcg::AssetManager::registerAsset(medium, "medium");
        component.medium_handle = medium->handle;
        return;
    }
}


}    // namespace Serialization

namespace GUI
{
void ComponentGUIRenderer<MediumComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                           Entity entity,
                                                           MediumComponent& component) const
{
#ifndef ATCG_HEADLESS
    MediumComponent copy = component;
    bool deactivated     = false;
    auto new_handle      = Utils::displayMediumSelection("medium", copy.medium_handle, deactivated);
    bool updated         = (new_handle != copy.medium_handle);
    copy.medium_handle   = new_handle;

    if(updated)
    {
        RevisionStack::startRecording<ComponentEditedRevision<MediumComponent>>(scene, entity);
        component = copy;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(MediumComponent);
}    // namespace atcg