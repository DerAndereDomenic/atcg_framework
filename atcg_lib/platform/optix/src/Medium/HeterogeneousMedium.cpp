#include <Medium/HeterogeneousMedium.h>

#include <Renderer/Texture.h>

#include <Scene/ComponentRegistry.h>

namespace atcg
{

using GridComponent = HeterogeneousMediumComponent::GridComponent;

HeterogeneousMedium::HeterogeneousMedium(const Dictionary& dict) : Medium(dict)
{
    HeterogeneousMediumData data;

    glm::mat4 to_world       = dict.getValueOr<glm::mat4>("to_world", glm::mat4(1));
    glm::mat4 world_to_local = glm::inverse(to_world);

    auto density_grid     = dict.getValue<GridComponent>("density_grid");
    auto density_texture  = AssetManager::getAsset<Texture3D>(density_grid.handle)->clone();
    _density_texture      = std::static_pointer_cast<Texture3D>(density_texture);
    auto emission_grid    = dict.getValue<GridComponent>("emission_grid");
    auto emission_texture = AssetManager::getAsset<Texture3D>(emission_grid.handle);
    _emission_texture     = emission_texture ? std::static_pointer_cast<Texture3D>(emission_texture->clone()) : nullptr;
    auto albedo_grid      = dict.getValue<GridComponent>("albedo_grid");
    auto albedo_texture   = AssetManager::getAsset<Texture3D>(albedo_grid.handle);
    _albedo_texture       = albedo_texture ? std::static_pointer_cast<Texture3D>(albedo_texture->clone()) : nullptr;

    data.density_grid.texture = _density_texture->getTextureObject();
    data.density_grid.scale   = density_grid.scale;
    data.density_majorant     = _density_texture->getData(atcg::GPU).max().item<float>() * data.density_grid.scale;
    {
        glm::mat4 to_uvw         = glm::mat4(1);
        glm::vec3 scale          = density_grid.bbox.max - density_grid.bbox.min;
        to_uvw                   = to_uvw * glm::scale(1.0f / scale);
        to_uvw                   = to_uvw * glm::translate(-density_grid.bbox.min);
        to_uvw                   = to_uvw * world_to_local;
        data.density_grid.to_uvw = to_uvw;
    }

    data.emission_grid.texture       = _emission_texture ? _emission_texture->getTextureObject() : 0;
    data.emission_grid.default_value = glm::vec3(0);
    data.emission_grid.scale         = emission_grid.scale;
    {
        glm::mat4 to_uvw          = glm::mat4(1);
        glm::vec3 scale           = emission_grid.bbox.max - emission_grid.bbox.min;
        to_uvw                    = to_uvw * glm::scale(1.0f / scale);
        to_uvw                    = to_uvw * glm::translate(-emission_grid.bbox.min);
        to_uvw                    = to_uvw * world_to_local;
        data.emission_grid.to_uvw = to_uvw;
    }

    data.albedo_grid.texture = _albedo_texture ? _albedo_texture->getTextureObject() : 0;
    data.albedo_grid.scale   = albedo_grid.scale;
    {
        glm::mat4 to_uvw        = glm::mat4(1);
        glm::vec3 scale         = albedo_grid.bbox.max - albedo_grid.bbox.min;
        to_uvw                  = to_uvw * glm::scale(1.0f / scale);
        to_uvw                  = to_uvw * glm::translate(-albedo_grid.bbox.min);
        to_uvw                  = to_uvw * world_to_local;
        data.albedo_grid.to_uvw = to_uvw;
    }

    _data_buffer.upload(&data);
}

HeterogeneousMedium::~HeterogeneousMedium()
{
    _density_texture->unmapDevicePointers();
    if(_albedo_texture) _albedo_texture->unmapDevicePointers();
    if(_emission_texture) _emission_texture->unmapDevicePointers();
}

void PipelineInitializer<HeterogeneousMedium>::apply(const atcg::ref_ptr<HeterogeneousMedium>& component) const
{
    // TODO
    // if(_phase_function != nullptr) _phase_function->ensureInitialized(pipeline, sbt);

    auto phase_function = component->getPhaseFunction();

    const std::string ptx_filename = "./bin/HeterogeneousMedium_ptx.ptx";
    OptixProgramGroup eval_transmittance_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__heterogeneousMedium_evalTransmittance"});
    OptixProgramGroup sample_medium_event_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__heterogeneousMedium_sampleMediumEvent"});

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

    j_density_grid["grid"]  = (uint64_t)component.density_grid.handle;
    j_density_grid["scale"] = component.density_grid.scale;
    nlohmann::json j_bbox_density;
    j_bbox_density["min"] = nlohmann::json::array(
        {component.density_grid.bbox.min.x, component.density_grid.bbox.min.y, component.density_grid.bbox.min.z});
    j_bbox_density["max"] = nlohmann::json::array(
        {component.density_grid.bbox.max.x, component.density_grid.bbox.max.y, component.density_grid.bbox.max.z});
    j_density_grid["bbox"] = j_bbox_density;

    j_albedo_grid["grid"]  = (uint64_t)component.albedo_grid.handle;
    j_albedo_grid["scale"] = component.albedo_grid.scale;
    nlohmann::json j_bbox_albedo;
    j_bbox_albedo["min"] = nlohmann::json::array(
        {component.albedo_grid.bbox.min.x, component.albedo_grid.bbox.min.y, component.albedo_grid.bbox.min.z});
    j_bbox_albedo["max"] = nlohmann::json::array(
        {component.albedo_grid.bbox.max.x, component.albedo_grid.bbox.max.y, component.albedo_grid.bbox.max.z});
    j_albedo_grid["bbox"] = j_bbox_albedo;

    j_emission_grid["grid"]  = (uint64_t)component.emission_grid.handle;
    j_emission_grid["scale"] = component.emission_grid.scale;
    nlohmann::json j_bbox_emission;
    j_bbox_emission["min"] = nlohmann::json::array(
        {component.emission_grid.bbox.min.x, component.emission_grid.bbox.min.y, component.emission_grid.bbox.min.z});
    j_bbox_emission["max"] = nlohmann::json::array(
        {component.emission_grid.bbox.max.x, component.emission_grid.bbox.max.y, component.emission_grid.bbox.max.z});
    j_emission_grid["bbox"] = j_bbox_emission;

    j["Heterogeneous Medium"]["density_grid"]  = j_density_grid;
    j["Heterogeneous Medium"]["albedo_grid"]   = j_albedo_grid;
    j["Heterogeneous Medium"]["emission_grid"] = j_emission_grid;
    j["Heterogeneous Medium"]["g"]             = component.g;
}


void ComponentSerializer<HeterogeneousMediumComponent>::deserialize_component(const std::string& file_path,
                                                                              const atcg::ref_ptr<Scene>& scene,
                                                                              Entity entity,
                                                                              nlohmann::json& j) const
{
    if(!j.contains("Heterogeneous Medium"))
    {
        return;
    }

    auto& component = entity.addComponent<HeterogeneousMediumComponent>();

    // --- Density grid ---
    const auto& j_density         = j["Heterogeneous Medium"]["density_grid"];
    component.density_grid.handle = (AssetHandle)j_density["grid"].get<uint64_t>();
    component.density_grid.scale  = j_density["scale"].get<float>();

    const auto& j_bbox_density      = j_density["bbox"];
    component.density_grid.bbox.min = glm::vec3(j_bbox_density["min"][0].get<float>(),
                                                j_bbox_density["min"][1].get<float>(),
                                                j_bbox_density["min"][2].get<float>());
    component.density_grid.bbox.max = glm::vec3(j_bbox_density["max"][0].get<float>(),
                                                j_bbox_density["max"][1].get<float>(),
                                                j_bbox_density["max"][2].get<float>());

    // --- Albedo grid ---
    const auto& j_albedo         = j["Heterogeneous Medium"]["albedo_grid"];
    component.albedo_grid.handle = (AssetHandle)j_albedo["grid"].get<uint64_t>();
    component.albedo_grid.scale  = j_albedo["scale"].get<float>();

    const auto& j_bbox_albedo      = j_albedo["bbox"];
    component.albedo_grid.bbox.min = glm::vec3(j_bbox_albedo["min"][0].get<float>(),
                                               j_bbox_albedo["min"][1].get<float>(),
                                               j_bbox_albedo["min"][2].get<float>());
    component.albedo_grid.bbox.max = glm::vec3(j_bbox_albedo["max"][0].get<float>(),
                                               j_bbox_albedo["max"][1].get<float>(),
                                               j_bbox_albedo["max"][2].get<float>());

    // --- Emission grid ---
    const auto& j_emission         = j["Heterogeneous Medium"]["emission_grid"];
    component.emission_grid.handle = (AssetHandle)j_emission["grid"].get<uint64_t>();
    component.emission_grid.scale  = j_emission["scale"].get<float>();

    const auto& j_bbox_emission      = j_emission["bbox"];
    component.emission_grid.bbox.min = glm::vec3(j_bbox_emission["min"][0].get<float>(),
                                                 j_bbox_emission["min"][1].get<float>(),
                                                 j_bbox_emission["min"][2].get<float>());
    component.emission_grid.bbox.max = glm::vec3(j_bbox_emission["max"][0].get<float>(),
                                                 j_bbox_emission["max"][1].get<float>(),
                                                 j_bbox_emission["max"][2].get<float>());

    component.g = j["Heterogeneous Medium"]["g"].get<float>();
}
}    // namespace Serialization

ATCG_REGISTER_COMPONENT_DRAW(HeterogeneousMediumComponent);
ATCG_REGISTER_COMPONENT_STORE(HeterogeneousMediumComponent);
ATCG_REGISTER_COMPONENT_SERIALIZATION(HeterogeneousMediumComponent);
}    // namespace atcg