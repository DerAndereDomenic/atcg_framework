#include <Scene/Components/PointSphereRenderComponent.h>
#include <Scene/ComponentRegistry.h>
#include <Utils/Utils.h>

namespace atcg
{

void ComponentRenderer<PointSphereRenderComponent>::renderComponent(atcg::RendererSystem* _renderer,
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

    // Actual rendering of component
    PointSphereRenderComponent renderer = entity.getComponent<PointSphereRenderComponent>();

    auto scene = entity.scene();

    atcg::ref_ptr<atcg::Shader> shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader", renderer.shader());

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto skybox     = auxiliary.getValueOr<atcg::ref_ptr<Skybox>>("skybox", AssetManager::getDummySkybox());
    auto has_skybox = auxiliary.getValueOr<bool>("has_skybox", false);

    if(renderer.visible)
    {
        uint32_t id          = Utils::setLights(_renderer, scene, point_light_depth_maps, shader);
        auto [ir_id, pre_id] = Utils::setSkyLight(_renderer, shader, skybox);
        shader->setInt("use_ibl", has_skybox);
        shader->setInt("entityID", entity.entity_handle());
        shader->setFloat("point_size", renderer.point_size);
        shader->setVec3("flat_color", glm::vec3(1));
        renderer.material()->uploadMaterial(_renderer, shader);
        uint32_t lut_id = _renderer->popTextureID();
        shader->setInt("lut", lut_id);
        GraphicsCommand::bindTexture(lut_id, AssetManager::getLUTTexture());

        auto vbo = geometry.graph()->getVerticesBuffer();

        auto sphere_mesh = AssetManager::getSphereMesh();
        auto vao_sphere  = sphere_mesh->getVerticesArray();

        vao_sphere->pushInstanceBuffer(vbo);

        GraphicsPipeline pipeline = GraphicsPipeline().setShader(shader);

        // _renderer->setPointSize(renderer.point_size);
        _renderer->drawVAO(vao_sphere,
                           camera,
                           transform.getModel(),
                           pipeline,
                           sphere_mesh->n_vertices(),
                           geometry.graph()->n_vertices());
        vao_sphere->popVertexBuffer();
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

namespace GUI
{
void ComponentGUIRenderer<PointSphereRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                      Entity entity,
                                                                      PointSphereRenderComponent& _component) const
{
#ifndef ATCG_HEADLESS
    PointSphereRenderComponent component = _component;
    std::string id                       = std::to_string(entity.getComponent<IDComponent>().ID());

    bool updated = ImGui::Checkbox("Visible##visiblepointsphere", &component.visible);

    float point_size = component.point_size;
    std::stringstream label;
    label << "Point Size##pointsphere" << id;
    if(ImGui::DragFloat(label.str().c_str(), &point_size, 0.001f, 0.001f, FLT_MAX / INT_MAX))
    {
        component.point_size = point_size;
        updated              = true;
    }

    // Material
    auto material_handle = component.material_handle;

    auto new_handle           = displayMaterialSelection("pointsphere", material_handle);
    updated                   = (new_handle != material_handle) || updated;
    component.material_handle = new_handle;

    auto shader_handle      = component.shader_handle;
    new_handle              = displayShaderSelection("pointsphere", shader_handle);
    updated                 = (new_handle != shader_handle) || updated;
    component.shader_handle = new_handle;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<PointSphereRenderComponent>>(scene, entity);
        _component = component;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(PointSphereRenderComponent);
}    // namespace atcg