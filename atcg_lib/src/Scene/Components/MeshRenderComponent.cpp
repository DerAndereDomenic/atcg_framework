#include <Scene/Components/MeshRenderComponent.h>
#include <Scene/ComponentRegistry.h>
#include <Utils/Utils.h>

#define MESH_RENDERER_KEY   "MeshRenderer"
#define SHADER_KEY          "Shader"
#define MATERIAL_KEY        "Material"
#define RECEIVE_SHADOWS_KEY "ReceiveShadow"

namespace atcg
{

void ComponentRenderer<MeshRenderComponent>::renderComponent(atcg::RendererSystem* _renderer,
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
    MeshRenderComponent renderer = entity.getComponent<MeshRenderComponent>();

    if(renderer.material()->getMaterialType() == MaterialType::MATERIAL_TYPE_NULL)
    {
        return;
    }

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
        shader->setInt("receive_shadow", (int)renderer.receive_shadow);
        shader->setInt("entityID", entity.entity_handle());
        shader->setVec3("flat_color", glm::vec3(1));
        renderer.material()->uploadMaterial(_renderer, shader);
        uint32_t lut_id = _renderer->popTextureID();
        shader->setInt("lut", lut_id);
        GraphicsCommand::bindTexture(lut_id, AssetManager::getLUTTexture());

        GraphicsPipeline pipeline = GraphicsPipeline().setShader(shader).setRasterizerState(
            RasterizerState().setCullMode(CullMode::ATCG_BACK_FACE_CULLING).enableCulling(true).enableCulling(true));

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
        renderer.material()->releaseTextureIDs(_renderer);
    }
}

namespace Serialization
{
void ComponentSerializer<MeshRenderComponent>::serialize_component(const std::string& file_path,
                                                                   const atcg::ref_ptr<Scene>& scene,
                                                                   Entity entity,
                                                                   MeshRenderComponent& component,
                                                                   nlohmann::json& j) const
{
    if(AssetManager::isAssetHandleValid(component.shader_handle))
    {
        j[MESH_RENDERER_KEY][SHADER_KEY] = (uint64_t)component.shader_handle;
    }
    j[MESH_RENDERER_KEY][RECEIVE_SHADOWS_KEY] = component.receive_shadow;

    j[MESH_RENDERER_KEY][MATERIAL_KEY] = (uint64_t)component.material_handle;
}

void ComponentSerializer<MeshRenderComponent>::deserialize_component(const std::string& file_path,
                                                                     const atcg::ref_ptr<Scene>& scene,
                                                                     Entity entity,
                                                                     nlohmann::json& j) const
{
    if(!j.contains(MESH_RENDERER_KEY))
    {
        return;
    }

    auto& renderer        = j[MESH_RENDERER_KEY];
    auto& renderComponent = entity.addComponent<MeshRenderComponent>();

    if(j[MESH_RENDERER_KEY].contains(SHADER_KEY))
    {
        renderComponent.shader_handle = (AssetHandle)j[MESH_RENDERER_KEY][SHADER_KEY];
    }


    if(renderer.contains(MATERIAL_KEY))
    {
        renderComponent.material_handle = (AssetHandle)renderer[MATERIAL_KEY];
    }

    renderComponent.receive_shadow = renderer.value(RECEIVE_SHADOWS_KEY, true);
}
}    // namespace Serialization

namespace GUI
{
void ComponentGUIRenderer<MeshRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               MeshRenderComponent& component) const
{
#ifndef ATCG_HEADLESS
    MeshRenderComponent component_copy = component;
    bool updated                       = ImGui::Checkbox("Visible##visiblemesh", &component_copy.visible);

    // Material
    auto material_handle = component_copy.material_handle;
    auto shader_handle   = component_copy.shader_handle;

    auto new_handle                = Utils::displayMaterialSelection("mesh", material_handle);
    updated                        = (new_handle != material_handle) || updated;
    component_copy.material_handle = new_handle;

    new_handle                   = Utils::displayShaderSelection("mesh", shader_handle);
    updated                      = (new_handle != shader_handle) || updated;
    component_copy.shader_handle = new_handle;

    updated = ImGui::Checkbox("Receive Shadows##MeshRenderComponent", &component_copy.receive_shadow) || updated;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<MeshRenderComponent>>(scene, entity);
        component = component_copy;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(MeshRenderComponent);
}    // namespace atcg