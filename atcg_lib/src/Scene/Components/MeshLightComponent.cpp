#include <Scene/Components/MeshLightComponent.h>
#include <Scene/ComponentRegistry.h>
#include <Scene/Components/TransformComponent.h>
#include <Scene/Components/GeometryComponent.h>
#include <Utils/Utils.h>

#define MESH_LIGHT_KEY       "MeshLight"
#define EMISSIVE_SCALE_KEY   "EmissiveScale"
#define EMISSIVE_COLOR_KEY   "EmissiveColor"
#define EMISSIVE_TEXTURE_KEY "EmissiveTexture"

namespace atcg
{
void ComponentRenderer<MeshLightComponent>::renderComponent(atcg::RendererSystem* _renderer,
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

namespace Serialization
{
void ComponentSerializer<MeshLightComponent>::serialize_component(const std::string& file_path,
                                                                  const atcg::ref_ptr<Scene>& scene,
                                                                  Entity entity,
                                                                  MeshLightComponent& component,
                                                                  nlohmann::json& j) const
{
    j[MESH_LIGHT_KEY][EMISSIVE_SCALE_KEY] = component.intensity;

    if(AssetManager::isAssetHandleValid(component.emissive_handle))
    {
        j[MESH_LIGHT_KEY][EMISSIVE_TEXTURE_KEY] = (uint64_t)component.emissive_handle;
    }
    else
    {
        auto data         = component.getEmissiveTexture()->getData(atcg::CPU);
        glm::u8vec3 color = {data.index({0, 0, 0}).item<uint8_t>(),
                             data.index({0, 0, 1}).item<uint8_t>(),
                             data.index({0, 0, 2}).item<uint8_t>()};

        glm::vec3 c(color);
        c = c / 255.0f;

        j[MESH_LIGHT_KEY][EMISSIVE_COLOR_KEY] = nlohmann::json::array({c.x, c.y, c.z});
    }
}

void ComponentSerializer<MeshLightComponent>::deserialize_component(const std::string& file_path,
                                                                    const atcg::ref_ptr<Scene>& scene,
                                                                    Entity entity,
                                                                    nlohmann::json& j) const
{
    if(!j.contains(MESH_LIGHT_KEY))
    {
        return;
    }

    auto& renderComponent = entity.addComponent<MeshLightComponent>();

    renderComponent.intensity = j[MESH_LIGHT_KEY].value(EMISSIVE_SCALE_KEY, 1.0f);
    if(j[MESH_LIGHT_KEY].contains(EMISSIVE_COLOR_KEY))
    {
        std::vector<float> emissive_color = j[MESH_LIGHT_KEY][EMISSIVE_COLOR_KEY];
        renderComponent.setEmissiveColor(glm::make_vec3(emissive_color.data()));
    }
    else if(j[MESH_LIGHT_KEY].contains(EMISSIVE_TEXTURE_KEY))
    {
        AssetHandle emissive_handle     = (AssetHandle)j[MESH_LIGHT_KEY][EMISSIVE_TEXTURE_KEY];
        renderComponent.emissive_handle = emissive_handle;
    }
}

}    // namespace Serialization


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

        auto new_handle = Utils::displayTexture2DSelection("meshlight", component.emissive_handle);

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