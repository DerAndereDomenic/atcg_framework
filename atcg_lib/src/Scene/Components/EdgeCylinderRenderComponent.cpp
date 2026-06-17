#include <Scene/Components/EdgeCylinderRenderComponent.h>
#include <Scene/ComponentRegistry.h>
#include <Utils/Utils.h>

#define EDGE_CYLINDER_RENDERER_KEY "EdgeCylinderRenderer"
#define MATERIAL_KEY               "Material"
#define RADIUS_KEY                 "Radius"
#define CULL_MODE_KEY              "CullMode"

namespace atcg
{

void ComponentRenderer<EdgeCylinderRenderComponent>::renderComponent(atcg::RendererSystem* _renderer,
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


    uint32_t entity_id           = entity.entity_handle();
    TransformComponent transform = entity.getComponent<TransformComponent>();
    GeometryComponent geometry   = entity.getComponent<GeometryComponent>();

    if(!geometry.graph())
    {
        ATCG_WARN("Entity does have geometry component but mesh is empty");
        return;
    }

    geometry.graph()->unmapAllPointers();

    BoundingBox bbox = geometry.graph()->getBoundingBox();
    bbox             = Utils::transformBoundingBox(bbox, transform.getModel());

    if(!Utils::isVisible(camera, bbox))
    {
        return;
    }

    // Actual rendering of component
    EdgeCylinderRenderComponent renderer = entity.getComponent<EdgeCylinderRenderComponent>();

    auto scene = entity.scene();

    atcg::ref_ptr<atcg::Shader> shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader",
                                                    _renderer->getShaderManager()->getShader("cylinder_edge"));

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto skybox     = auxiliary.getValueOr<atcg::ref_ptr<Skybox>>("skybox", AssetManager::getDummySkybox());
    auto has_skybox = auxiliary.getValueOr<bool>("has_skybox", false);

    if(renderer.visible)
    {
        uint32_t id          = Utils::setLights(_renderer, scene, point_light_depth_maps, shader);
        auto [ir_id, pre_id] = Utils::setSkyLight(_renderer, shader, skybox);
        shader->setInt("use_ibl", has_skybox);
        shader->setFloat("edge_radius", renderer.radius);
        shader->setInt("entityID", entity.entity_handle());
        shader->setVec3("flat_color", glm::vec3(1));
        renderer.material()->uploadMaterial(_renderer, shader);
        uint32_t lut_id = _renderer->popTextureID();
        shader->setInt("lut", lut_id);
        GraphicsCommand::bindTexture(lut_id, AssetManager::getLUTTexture());

        auto points  = geometry.graph()->getVerticesBuffer();
        auto indices = geometry.graph()->getEdgesBuffer();

        GraphicsCommand::bindStorageBuffer(0, points);

        auto cylinder_mesh = AssetManager::getCylinderMesh();
        auto vao_cylinder  = cylinder_mesh->getVerticesArray();

        vao_cylinder->pushInstanceBuffer(indices);

        GraphicsPipeline pipeline = GraphicsPipeline().setShader(shader).setRasterizerState(
            RasterizerState().setCullMode(renderer.cull_mode).enableCulling(true).enableCulling(true));

        _renderer->drawVAO(vao_cylinder,
                           camera,
                           transform.getModel(),
                           pipeline,
                           cylinder_mesh->n_vertices(),
                           geometry.graph()->n_edges());
        vao_cylinder->popVertexBuffer();
        if(id != -1)
        {
            _renderer->pushTextureID(id);
        }
        if(ir_id != -1)
        {
            _renderer->pushTextureID(ir_id);
        }
        if(pre_id != -1)
        {
            _renderer->pushTextureID(pre_id);
        }
        if(lut_id != -1)
        {
            _renderer->pushTextureID(lut_id);
        }
        renderer.material()->releaseTextureIDs(_renderer);
    }
}

namespace Serialization
{
void ComponentSerializer<EdgeCylinderRenderComponent>::serialize_component(const std::string& file_path,
                                                                           const atcg::ref_ptr<Scene>& scene,
                                                                           Entity entity,
                                                                           EdgeCylinderRenderComponent& component,
                                                                           nlohmann::json& j) const
{
    j[EDGE_CYLINDER_RENDERER_KEY][RADIUS_KEY] = component.radius;

    j[EDGE_CYLINDER_RENDERER_KEY][MATERIAL_KEY]  = (uint64_t)component.material_handle;
    j[EDGE_CYLINDER_RENDERER_KEY][CULL_MODE_KEY] = (int)component.cull_mode;
}

void ComponentSerializer<EdgeCylinderRenderComponent>::deserialize_component(const std::string& file_path,
                                                                             const atcg::ref_ptr<Scene>& scene,
                                                                             Entity entity,
                                                                             nlohmann::json& j) const
{
    if(!j.contains(EDGE_CYLINDER_RENDERER_KEY))
    {
        return;
    }

    auto& renderer         = j[EDGE_CYLINDER_RENDERER_KEY];
    auto& renderComponent  = entity.addComponent<EdgeCylinderRenderComponent>();
    renderComponent.radius = renderer.value(RADIUS_KEY, 0.001f);


    if(renderer.contains(MATERIAL_KEY))
    {
        renderComponent.material_handle = (AssetHandle)renderer[MATERIAL_KEY];
    }

    renderComponent.cull_mode =
        (atcg::CullMode)renderer.value(CULL_MODE_KEY, (int)atcg::CullMode::ATCG_BACK_FACE_CULLING);
}

}    // namespace Serialization


namespace GUI
{
void ComponentGUIRenderer<EdgeCylinderRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                       Entity entity,
                                                                       EdgeCylinderRenderComponent& _component) const
{
#ifndef ATCG_HEADLESS
    EdgeCylinderRenderComponent component = _component;
    std::string id                        = std::to_string(entity.getComponent<IDComponent>().ID());

    bool updated = ImGui::Checkbox("Visible##visibleedgecylinder", &component.visible);
    std::stringstream label;
    label << "Radius##edgecylinder" << id;
    float radius = component.radius;
    if(ImGui::DragFloat(label.str().c_str(), &radius, 0.001f, 0.001f, FLT_MAX / INT_MAX))
    {
        component.radius = radius;
        updated          = true;
    }

    // Material
    auto material_handle = component.material_handle;

    auto new_handle           = Utils::displayMaterialSelection("edgecylinder", material_handle);
    updated                   = (new_handle != material_handle) || updated;
    component.material_handle = new_handle;

    auto new_cull_mode  = Utils::displayCullModeSelection("edgecylinder", component.cull_mode);
    updated             = (new_cull_mode != component.cull_mode) || updated;
    component.cull_mode = new_cull_mode;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<EdgeCylinderRenderComponent>>(scene, entity);
        _component = component;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(EdgeCylinderRenderComponent);
}    // namespace atcg