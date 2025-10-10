#include <Medium/HomogeneousMedium.h>

#include <Scene/ComponentRegistry.h>

#ifndef ATCG_HEADLESS
    #include <imgui.h>
#endif

namespace atcg
{
HomogeneousMedium::HomogeneousMedium(const atcg::Dictionary& dict) : Medium(dict)
{
    HomogeneousMediumData data;
    data.sigma_a = dict.getValueOr<glm::vec3>("sigma_a", glm::vec3(0));
    data.sigma_s = dict.getValueOr<glm::vec3>("sigma_s", glm::vec3(0));
    data.Le      = glm::vec3(dict.getValueOr<glm::vec3>("Le", glm::vec3(0)));

    _data_buffer.upload(&data);
}

HomogeneousMedium::~HomogeneousMedium() {}

void PipelineInitializer<HomogeneousMedium>::apply(const atcg::ref_ptr<HomogeneousMedium>& component) const
{
    // TODO
    // if(_phase_function != nullptr) _phase_function->ensureInitialized(pipeline, sbt);

    auto phase_function = component->getPhaseFunction();

    const std::string ptx_filename = "./bin/HomogeneousMedium_ptx.ptx";
    OptixProgramGroup eval_transmittance_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__homogeneousMedium_evalTransmittance"});
    OptixProgramGroup sample_medium_event_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__homogeneousMedium_sampleMediumEvent"});

    uint32_t eval_transmittance_index =
        sbt->addCallableEntry(eval_transmittance_prog_group, component->getDataBuffer().get());
    uint32_t sample_medium_event_index =
        sbt->addCallableEntry(sample_medium_event_prog_group, component->getDataBuffer().get());

    MediumVPtrTable vptr_table_data;
    vptr_table_data.evalCallIndex   = eval_transmittance_index;
    vptr_table_data.sampleCallIndex = sample_medium_event_index;
    vptr_table_data.phase_function  = phase_function ? phase_function->getVPtrTable() : nullptr;

    component->getVPtrTableHolder().upload(&vptr_table_data);

    component->markInitialized();
}

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

    j["Homogeneous Medium"]["albedo"]   = nlohmann::json::array({albedo.x, albedo.y, albedo.z});
    j["Homogeneous Medium"]["density"]  = density;
    j["Homogeneous Medium"]["g"]        = g;
    j["Homogeneous Medium"]["Le"]       = Le;
    j["Homogeneous Medium"]["Le_color"] = nlohmann::json::array({Le_color.x, Le_color.y, Le_color.z});
}


void ComponentSerializer<HomogeneousMediumComponent>::deserialize_component(const std::string& file_path,
                                                                            const atcg::ref_ptr<Scene>& scene,
                                                                            Entity entity,
                                                                            nlohmann::json& j) const
{
    if(!j.contains("Homogeneous Medium"))
    {
        return;
    }

    std::vector<float> albedo   = j["Homogeneous Medium"].value("albedo", std::vector<float> {1.0f, 1.0f, 1.0f});
    std::vector<float> Le_color = j["Homogeneous Medium"].value("Le_color", std::vector<float> {1.0f, 1.0f, 1.0f});

    auto& component    = entity.addComponent<HomogeneousMediumComponent>();
    component.albedo   = glm::make_vec3(albedo.data());
    component.g        = j["Homogeneous Medium"]["g"];
    component.Le       = j["Homogeneous Medium"]["Le"];
    component.Le_color = glm::make_vec3(Le_color.data());
    component.density  = j["Homogeneous Medium"]["density"];
}
}    // namespace Serialization

ATCG_REGISTER_COMPONENT_DRAW(HomogeneousMediumComponent);
ATCG_REGISTER_COMPONENT_STORE(HomogeneousMediumComponent);
ATCG_REGISTER_COMPONENT_SERIALIZATION(HomogeneousMediumComponent);

}    // namespace atcg