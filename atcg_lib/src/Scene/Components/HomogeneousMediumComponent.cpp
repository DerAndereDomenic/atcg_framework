#include <Scene/Components/HomogeneousMediumComponent.h>
#include <Scene/ComponentRegistry.h>
#include <Scene/Components/TransformComponent.h>
#include <Scene/Components/GeometryComponent.h>
#include <Scene/Components/MeshRenderComponent.h>

#define HOMOGENEOUS_MEDIUM_KEY "Homogeneous Medium"
#define ALBEDO_KEY             "albedo"
#define DENSITY_KEY            "density"
#define G_KEY                  "g"
#define LE_KEY                 "Le"
#define LE_COLOR_KEY           "Le_color"

namespace atcg
{

void ComponentRenderer<HomogeneousMediumComponent>::renderComponent(atcg::RendererSystem* _renderer,
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
    HomogeneousMediumComponent& medium = entity.getComponent<HomogeneousMediumComponent>();
    MeshRenderComponent& mesh_renderer = entity.getComponent<MeshRenderComponent>();

    if(mesh_renderer.material()->getMaterialType() != MaterialType::MATERIAL_TYPE_NULL)
    {
        return;
    }

    auto scene = entity.scene();

    atcg::ref_ptr<atcg::Shader> shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader",
                                                    _renderer->getShaderManager()->getShader("volume_hom"));

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
        shader->setVec3("albedo", medium.albedo);
        shader->setFloat("density", medium.density);
        shader->setInt("entityID", entity.entity_handle());
        shader->setMat4("invView", glm::inverse(camera->getView()));
        shader->setMat4("invProj", glm::inverse(camera->getProjection()));
        shader->setFloat("g", medium.g);
        shader->setFloat("Le", medium.Le);
        shader->setVec3("Le_color", medium.Le_color);

        GraphicsPipeline pipeline = GraphicsPipeline().setShader(shader).setRasterizerState(
            RasterizerState().enableCulling(true).setCullMode(CullMode::ATCG_BACK_FACE_CULLING));

        _renderer->drawVAO(geometry.graph()->getVerticesArray(),
                           camera,
                           transform.getModel(),
                           pipeline,
                           geometry.graph()->n_vertices());

        _renderer->pushTextureID(id);
    }
}

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

    bool updated     = false;
    bool deactivated = false;
    updated          = ImGui::DragFloat("Density##homogen", &_component.density, 0.05f, 0.0f, 50.0f) || updated;
    deactivated      = ImGui::IsItemDeactivated() || deactivated;
    updated          = ImGui::DragFloat("g##homogen", &_component.g, 0.01f, -1.0f, 1.0f) || updated;
    deactivated      = ImGui::IsItemDeactivated() || deactivated;
    updated          = ImGui::ColorEdit3("albedo##homogen", glm::value_ptr(_component.albedo)) || updated;
    deactivated      = ImGui::IsItemDeactivated() || deactivated;
    updated          = ImGui::DragFloat("Le##homogen", &_component.Le, 0.01f, 0.0f, 100.0f) || updated;
    deactivated      = ImGui::IsItemDeactivated() || deactivated;
    updated          = ImGui::ColorEdit3("LeColor##homogen", glm::value_ptr(_component.Le_color)) || updated;
    deactivated      = ImGui::IsItemDeactivated() || deactivated;

    if(updated && !atcg::RevisionStack::isRecording())
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<HomogeneousMediumComponent>>(scene, entity);
    }

    if(updated)
    {
        component = _component;
    }

    if(deactivated && atcg::RevisionStack::isRecording())
    {
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(HomogeneousMediumComponent);
}    // namespace atcg