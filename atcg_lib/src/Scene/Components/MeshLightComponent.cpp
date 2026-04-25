#include <Scene/Components/MeshLightComponent.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{
void ComponentRenderer<MeshLightComponent>::renderComponent(atcg::RendererSystem* _renderer,
                                                            Entity entity,
                                                            const atcg::ref_ptr<Camera>& camera,
                                                            atcg::Dictionary& auxiliary) const
{
    uint32_t entity_id           = entity.entity_handle();
    TransformComponent transform = entity.getComponent<TransformComponent>();
    GeometryComponent geometry   = entity.getComponent<GeometryComponent>();

    // Actual rendering of component
    MeshLightComponent renderer = entity.getComponent<MeshLightComponent>();

    auto scene = entity.scene();

    atcg::ref_ptr<atcg::Shader> shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader",
                                                    _renderer->getShaderManager()->getShader("emissive"));

    // if(renderer.visible)
    {
        auto emissive_id = _renderer->popTextureID();
        GraphicsCommand::bindTexture(emissive_id, renderer.getEmissiveTexture());
        shader->setInt("texture_emissive", emissive_id);
        shader->setFloat("emissive_scaling", renderer.intensity);
        shader->setInt("entityID", entity.entity_handle());
        GraphicsPipeline pipeline = GraphicsPipeline().setShader(shader);
        _renderer->drawVAO(geometry.graph()->getVerticesArray(),
                           camera,
                           transform.getModel(),
                           pipeline,
                           geometry.graph()->n_vertices());

        _renderer->pushTextureID(emissive_id);
    }
}

namespace GUI
{
void ComponentGUIRenderer<MeshLightComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                              Entity entity,
                                                              MeshLightComponent& _component) const
{
#ifndef ATCG_HEADLESS
    MeshLightComponent component = _component;

    float updated = false;
    {
        // auto spec        = component.getEmissiveTexture()->getSpecification();
        // bool useTextures = spec.width != 1 || spec.height != 1;

        updated = ImGui::DragFloat("Scaling", &component.intensity, 0.005f, 0.0f, FLT_MAX) || updated;

        if(!AssetManager::isAssetHandleValid(component.emissive_handle))
        {
            auto emissive = component.getEmissiveTexture()->getData(atcg::CPU);

            float color[4] = {emissive.index({0, 0, 0}).item<float>() / 255.0f,
                              emissive.index({0, 0, 1}).item<float>() / 255.0f,
                              emissive.index({0, 0, 2}).item<float>() / 255.0f,
                              emissive.index({0, 0, 3}).item<float>() / 255.0f};

            if(ImGui::ColorEdit4("Emissive##mesh_light", color))
            {
                glm::vec4 new_color = glm::make_vec4(color);
                component.setEmissiveColor(new_color);
                updated = true;
            }
        }

        ImGui::Separator();

        auto new_handle = displayTexture2DSelection("meshlight", component.emissive_handle);

        updated                   = (new_handle != component.emissive_handle) || updated;
        component.emissive_handle = new_handle;

        ImGui::Separator();
    }

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<MeshLightComponent>>(scene, entity);
        _component = component;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(MeshLightComponent);
}    // namespace atcg