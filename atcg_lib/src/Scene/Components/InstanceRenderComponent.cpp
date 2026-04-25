#include <Scene/Components/InstanceRenderComponent.h>
#include <Scene/ComponentRegistry.h>
#include <Utils/Utils.h>

namespace atcg
{

void ComponentRenderer<InstanceRenderComponent>::renderComponent(atcg::RendererSystem* _renderer,
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
    InstanceRenderComponent renderer = entity.getComponent<InstanceRenderComponent>();

    auto override_shader =
        renderer.shader() ? renderer.shader() : _renderer->getShaderManager()->getShader("instanced");

    atcg::ref_ptr<atcg::Shader> shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader", override_shader);

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto skybox     = auxiliary.getValueOr<atcg::ref_ptr<Skybox>>("skybox", AssetManager::getDummySkybox());
    auto has_skybox = auxiliary.getValueOr<bool>("has_skybox", false);

    auto scene = entity.scene();

    if(renderer.visible)
    {
        auto vao = geometry.graph()->getVerticesArray();
        for(int i = 0; i < renderer.instance_vbos.size(); ++i)
        {
            renderer.instance_vbos[i]->unmapPointers();
            vao->pushInstanceBuffer(renderer.instance_vbos[i]);
        }

        uint32_t id          = Utils::setLights(_renderer, scene, point_light_depth_maps, shader);
        auto [ir_id, pre_id] = Utils::setSkyLight(_renderer, shader, skybox);
        shader->setInt("use_ibl", has_skybox);
        shader->setInt("receive_shadow", (int)renderer.receive_shadow);
        shader->setInt("entityID", entity.entity_handle());
        shader->setVec3("flat_color", glm::vec3(1));
        renderer.material()->uploadMaterial(_renderer, shader);
        uint32_t lut_id = _renderer->popTextureID();
        shader->setInt("lut", lut_id);
        GraphicsCommand::bindTexture(lut_id, AssetManager::getLUTTexture());

        auto instance_vbo    = vao->peekVertexBuffer();
        uint32_t n_instances = instance_vbo->size() / instance_vbo->getLayout().getStride();

        GraphicsPipeline pipeline = GraphicsPipeline().setShader(shader);

        _renderer->drawVAO(vao, camera, transform.getModel(), pipeline, geometry.graph()->n_vertices(), n_instances);
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

        for(int i = 0; i < renderer.instance_vbos.size(); ++i)
        {
            vao->popVertexBuffer();
        }
    }
}

namespace GUI
{
void ComponentGUIRenderer<InstanceRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                   Entity entity,
                                                                   InstanceRenderComponent& _component) const
{
#ifndef ATCG_HEADLESS
    InstanceRenderComponent component = _component;
    bool updated                      = ImGui::Checkbox("Visible##visibleinstance", &component.visible);

    // Material
    auto material_handle = component.material_handle;

    auto new_handle           = displayMaterialSelection("instance", material_handle);
    updated                   = (new_handle != material_handle) || updated;
    component.material_handle = new_handle;
    updated = ImGui::Checkbox("Receive Shadows##InstanceRenderComponent", &component.receive_shadow) || updated;

    auto shader_handle      = component.shader_handle;
    new_handle              = displayShaderSelection("instance", shader_handle);
    updated                 = (new_handle != shader_handle) || updated;
    component.shader_handle = new_handle;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<InstanceRenderComponent>>(scene, entity);
        _component = component;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(InstanceRenderComponent);
}    // namespace atcg