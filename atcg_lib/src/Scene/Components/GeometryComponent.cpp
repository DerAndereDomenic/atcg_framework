#include <Scene/Components/GeometryComponent.h>
#include <Scene/ComponentRegistry.h>
#include <Scene/Components/TransformComponent.h>
#include <Utils/Utils.h>

#define GEOMETRY_KEY          "Geometry"
#define DRAW_BOUNDING_BOX_KEY "DrawBoundingBox"

namespace atcg
{

void ComponentRenderer<GeometryComponent>::renderComponent(atcg::RendererSystem* _renderer,
                                                           Entity entity,
                                                           const atcg::ref_ptr<Camera>& camera,
                                                           atcg::Dictionary& auxiliary) const
{
    atcg::GeometryComponent& comp = entity.getComponent<GeometryComponent>();
    if(!comp.draw_bounding_box)
    {
        return;
    }

    if(!comp.graph())
    {
        return;
    }

    if(!entity.hasComponent<TransformComponent>())
    {
        return;
    }

    auto cube                 = AssetManager::getCubeMesh();
    auto shader               = _renderer->getShaderManager()->getShader("edge");
    GraphicsPipeline pipeline = GraphicsPipeline()
                                    .setShader(shader)
                                    .setPrimitiveTopology(PrimitiveTopology::ATCG_POINTS)
                                    .setRasterizerState(RasterizerState().enableCulling(false).setLineSize(2.0f));

    uint32_t entity_id = entity.entity_handle();
    shader->setInt("entityID", entity_id);
    shader->setVec3("flat_color", glm::vec3(0.0f, 1.0f, 0.0f));

    auto& transform = entity.getComponent<TransformComponent>();
    glm::mat4 model = transform.getModel();

    auto points = cube->getVerticesBuffer();
    GraphicsCommand::bindStorageBuffer(0, points);

    BoundingBox bbox = comp.graph()->getBoundingBox();
    bbox             = Utils::transformBoundingBox(bbox, model);
    model            = Utils::boundingBoxToModelMatrix(bbox);

    _renderer->drawVAO(cube->getEdgesArray(), camera, model, pipeline, cube->n_edges());
}

namespace Serialization
{
void ComponentSerializer<GeometryComponent>::serialize_component(const std::string& file_path,
                                                                 const atcg::ref_ptr<Scene>& scene,
                                                                 Entity entity,
                                                                 GeometryComponent& component,
                                                                 nlohmann::json& j) const
{
    j[GEOMETRY_KEY]          = (uint64_t)component.graph_handle;
    j[DRAW_BOUNDING_BOX_KEY] = component.draw_bounding_box;
}

void ComponentSerializer<GeometryComponent>::deserialize_component(const std::string& file_path,
                                                                   const atcg::ref_ptr<Scene>& scene,
                                                                   Entity entity,
                                                                   nlohmann::json& j) const
{
    if(!j.contains(GEOMETRY_KEY))
    {
        return;
    }

    auto& geometry             = entity.addComponent<GeometryComponent>();
    geometry.graph_handle      = (AssetHandle)j[GEOMETRY_KEY];
    geometry.draw_bounding_box = j.value(DRAW_BOUNDING_BOX_KEY, false);
}


}    // namespace Serialization

namespace GUI
{
void ComponentGUIRenderer<GeometryComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                             Entity entity,
                                                             GeometryComponent& component) const
{
#ifndef ATCG_HEADLESS
    GeometryComponent copy = component;
    auto new_handle        = Utils::displayGraphSelection("geometry", copy.graph_handle);
    bool updated           = (new_handle != copy.graph_handle);
    copy.graph_handle      = new_handle;

    updated = updated || ImGui::Checkbox("Draw Bounding Box", &copy.draw_bounding_box);

    if(updated)
    {
        RevisionStack::startRecording<ComponentEditedRevision<GeometryComponent>>(scene, entity);
        component = copy;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(GeometryComponent);
}    // namespace atcg