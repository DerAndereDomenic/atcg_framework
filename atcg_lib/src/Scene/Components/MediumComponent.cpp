#include <Scene/Components/MediumComponent.h>
#include <Scene/ComponentRegistry.h>
#include <Utils/Utils.h>

#include <Material/MediumRegistry.h>
#include <Material/PhaseFunctionRegistry.h>

#define MEDIUM_KEY         "Medium"
#define PHASE_FUNCTION_KEY "PhaseFunction"

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
    if(!entity.hasComponent<TransformComponent>())
    {
        ATCG_WARN("Entity does not have transform component!");
        return;
    }

    if(!entity.hasComponent<GeometryComponent>())
    {
        ATCG_WARN("Entity does not have geometry component!");
        return;
    }

    if(!entity.hasComponent<MeshRenderComponent>())
    {
        ATCG_WARN("Entity does not have mesh render component!");
        return;
    }

    uint32_t entity_id           = entity.entity_handle();
    TransformComponent transform = entity.getComponent<TransformComponent>();
    GeometryComponent geometry   = entity.getComponent<GeometryComponent>();

    if(!geometry.graph())
    {
        ATCG_WARN("Entity does have geometry component but mesh is empty");
        return;
    }

    geometry.graph()->unmapAllPointers();

    // Actual rendering of component
    MediumComponent& medium_component  = entity.getComponent<MediumComponent>();
    MeshRenderComponent& mesh_renderer = entity.getComponent<MeshRenderComponent>();

    if(!medium_component.medium())
    {
        ATCG_WARN("Entity has MediumComponent but medium is nullptr");
        return;
    }

    if(!medium_component.phase_function())
    {
        ATCG_WARN("Entity has MediumComponent but phase function is nullptr");
        return;
    }

    auto medium = medium_component.medium();
    auto phase  = medium_component.phase_function();

    if(!hasMaterialFlag(mesh_renderer.material()->flags(), MaterialFlag::NullTransmission))
    {
        return;
    }

    auto scene = entity.scene();

    auto medium_shader = hasMediumFlag(medium->flags(), MediumFlag::Heterogeneous)
                             ? _renderer->getShaderManager()->getShader("volume_het")
                             : _renderer->getShaderManager()->getShader("volume_hom");

    atcg::ref_ptr<atcg::Shader> shader = auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader", medium_shader);

    auto depth_map = auxiliary.getValueOr<atcg::ref_ptr<atcg::Texture2D>>("depth_map", nullptr);

    if(!depth_map)
    {
        ATCG_WARN("No depth map provided for volume rendering, skipping homogeneous medium component");
        return;
    }

    if(mesh_renderer.visible)
    {
        uint32_t id = _renderer->popTextureID();
        GraphicsCommand::bindTexture(id, depth_map);
        shader->setInt("back_depth", id);
        shader->setInt("entityID", entity.entity_handle());
        shader->setMat4("invView", glm::inverse(camera->getView()));
        shader->setMat4("invProj", glm::inverse(camera->getProjection()));
        phase->uploadPhaseFunction(_renderer, shader);

        medium->uploadMedium(_renderer, shader, transform.getModel());

        GraphicsPipeline pipeline = GraphicsPipeline().setShader(shader).setRasterizerState(
            RasterizerState().enableCulling(true).setCullMode(CullMode::ATCG_BACK_FACE_CULLING));

        _renderer->drawVAO(geometry.graph()->getVerticesArray(),
                           camera,
                           transform.getModel(),
                           pipeline,
                           geometry.graph()->n_vertices());

        _renderer->pushTextureID(id);
        medium->releaseTextureIDs(_renderer);
    }
}

namespace Serialization
{
void ComponentSerializer<MediumComponent>::serialize_component(const std::string& file_path,
                                                               const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               MediumComponent& component,
                                                               nlohmann::json& j) const
{
    j[MEDIUM_KEY]         = (uint64_t)component.medium_handle;
    j[PHASE_FUNCTION_KEY] = (uint64_t)component.phase_function_handle;
}

void ComponentSerializer<MediumComponent>::deserialize_component(const std::string& file_path,
                                                                 const atcg::ref_ptr<Scene>& scene,
                                                                 Entity entity,
                                                                 nlohmann::json& j) const
{
    if(j.contains(MEDIUM_KEY))
    {
        auto& medium                 = entity.addComponent<MediumComponent>();
        medium.medium_handle         = (AssetHandle)j[MEDIUM_KEY];
        medium.phase_function_handle = (AssetHandle)j[PHASE_FUNCTION_KEY];
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

        atcg::ref_ptr<PhaseFunction> phase_function =
            atcg::PhaseFunctionRegistry::deserializePhaseFunction("HenyeyGreenstein", file_path, medium_json);
        atcg::AssetManager::registerAsset(phase_function, "phase_function");

        component.medium_handle         = medium->handle;
        component.phase_function_handle = phase_function->handle;
        return;
    }

    if(j.contains(HETEROGENEOUS_MEDIUM_KEY))
    {
        auto medium_json = j[HETEROGENEOUS_MEDIUM_KEY];

        auto& component = entity.addComponent<MediumComponent>();

        auto medium = atcg::MediumRegistry::deserializeMedium("Heterogeneous", file_path, medium_json);
        atcg::AssetManager::registerAsset(medium, "medium");

        atcg::ref_ptr<PhaseFunction> phase_function =
            atcg::PhaseFunctionRegistry::deserializePhaseFunction("HenyeyGreenstein", file_path, medium_json);
        atcg::AssetManager::registerAsset(phase_function, "phase_function");

        component.medium_handle         = medium->handle;
        component.phase_function_handle = phase_function->handle;
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

    auto new_phase_handle =
        Utils::displayPhaseFunctionSelection("phase_function", copy.phase_function_handle, deactivated);
    updated                    = updated || (new_phase_handle != copy.phase_function_handle);
    copy.phase_function_handle = new_phase_handle;

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