#include <Scene/Components/EdgeRenderComponent.h>
#include <Scene/ComponentRegistry.h>
#include <Utils/Utils.h>

namespace atcg
{

void ComponentRenderer<EdgeRenderComponent>::renderComponent(atcg::RendererSystem* _renderer,
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
    EdgeRenderComponent renderer = entity.getComponent<EdgeRenderComponent>();

    auto scene = entity.scene();

    atcg::ref_ptr<atcg::Shader> shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader",
                                                    _renderer->getShaderManager()->getShader("edge"));

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
        shader->setVec3("flat_color", renderer.color);
        renderer.default_material->uploadMaterial(_renderer, shader);
        uint32_t lut_id = _renderer->popTextureID();
        shader->setInt("lut", lut_id);
        GraphicsCommand::bindTexture(lut_id, AssetManager::getLUTTexture());

        auto points = geometry.graph()->getVerticesBuffer();
        GraphicsCommand::bindStorageBuffer(0, points);

        GraphicsPipeline pipeline = GraphicsPipeline()
                                        .setShader(shader)
                                        .setRasterizerState(RasterizerState().setLineSize(1.0f))    // TODO
                                        .setPrimitiveTopology(PrimitiveTopology::ATCG_POINTS);

        _renderer->drawVAO(geometry.graph()->getEdgesArray(),
                           camera,
                           transform.getModel(),
                           pipeline,
                           geometry.graph()->n_edges());
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
        renderer.default_material->releaseTextureIDs(_renderer);
    }
}

namespace GUI
{
void ComponentGUIRenderer<EdgeRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               EdgeRenderComponent& _component) const
{
#ifndef ATCG_HEADLESS
    EdgeRenderComponent component = _component;

    std::string id = std::to_string(entity.getComponent<IDComponent>().ID());

    bool updated    = ImGui::Checkbox("Visible##visibleedge", &component.visible);
    glm::vec3 color = component.color;
    std::stringstream label;
    label << "Base Color##edge" << id;
    if(ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(color)))
    {
        component.color = color;
        updated         = true;
    }

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<EdgeRenderComponent>>(scene, entity);
        _component = component;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(EdgeRenderComponent);
}    // namespace atcg