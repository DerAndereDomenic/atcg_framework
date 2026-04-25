#include <Scene/Components/PointRenderComponent.h>
#include <Scene/ComponentRegistry.h>
#include <Utils/Utils.h>

namespace atcg
{

void ComponentRenderer<PointRenderComponent>::renderComponent(atcg::RendererSystem* _renderer,
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
    PointRenderComponent renderer = entity.getComponent<PointRenderComponent>();

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
        shader->setVec3("flat_color", renderer.color);
        renderer.default_material->uploadMaterial(_renderer, shader);
        uint32_t lut_id = _renderer->popTextureID();
        shader->setInt("lut", lut_id);
        GraphicsCommand::bindTexture(lut_id, AssetManager::getLUTTexture());

        GraphicsPipeline pipeline = GraphicsPipeline()
                                        .setShader(shader)
                                        .setRasterizerState(RasterizerState().setPointSize(renderer.point_size))
                                        .setPrimitiveTopology(PrimitiveTopology::ATCG_POINTS);

        _renderer->drawVAO(geometry.graph()->getVerticesArray(),
                           camera,
                           transform.getModel(),
                           pipeline,
                           geometry.graph()->n_vertices());
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
void ComponentGUIRenderer<PointRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                Entity entity,
                                                                PointRenderComponent& _component) const
{
#ifndef ATCG_HEADLESS
    std::string id = std::to_string(entity.getComponent<IDComponent>().ID());

    PointRenderComponent component = _component;

    bool updated    = ImGui::Checkbox("Visible##visiblepoints", &component.visible);
    glm::vec3 color = component.color;
    std::stringstream label;
    label << "Base Color##point" << id;
    if(ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(color)))
    {
        component.color = color;
        updated         = true;
    }

    int point_size = (int)component.point_size;
    label.str(std::string());
    label << "Point Size##point" << id;
    if(ImGui::DragInt(label.str().c_str(), &point_size, 1, 1, INT_MAX))
    {
        component.point_size = (float)point_size;
        updated              = true;
    }

    auto shader_handle      = component.shader_handle;
    auto new_handle         = displayShaderSelection("point", shader_handle);
    updated                 = (new_handle != shader_handle) || updated;
    component.shader_handle = new_handle;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<PointRenderComponent>>(scene, entity);
        _component = component;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(PointRenderComponent);
}    // namespace atcg