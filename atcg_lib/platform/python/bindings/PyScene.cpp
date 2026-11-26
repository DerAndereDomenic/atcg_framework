#include "Bindings.h"

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
// Entities, scenes, scene renderer, ray tracing queries and clipboard
void bindScene(BindingContext& ctx)
{
    py::module_ m = ctx.module();

    auto m_entity = py::class_<atcg::Entity>(m, "Entity");
    auto m_scene  = py::class_<atcg::Scene, atcg::Asset, atcg::ref_ptr<atcg::Scene>>(m, "Scene");

    auto m_scene_renderer = m.def_submodule("SceneRenderer");

    auto m_entity_handle = py::class_<entt::entity>(m, "EntityHandle");

    auto m_hit_info = py::class_<atcg::Tracing::HitInfo>(m, "HitInfo");

    auto m_clipboard = m.def_submodule("Clipboard");

    ctx.define(
        [=]() mutable
        {
            m_entity_handle.def(py::init<uint32_t>(), "handle"_a);

            m_entity.def(py::init<>())
                .def(py::init<entt::entity, atcg::Scene*>(), "handle"_a, "scene"_a)
                .def(py::init<>([](entt::entity e, const atcg::ref_ptr<atcg::Scene>& scene)
                                { return atcg::Entity(e, scene.get()); }),
                     "handle"_a,
                     "scene"_a)
                .def(
                    "addTransformComponent",
                    [](atcg::Entity& entity,
                       const glm::vec3& position,
                       const glm::vec3& scale,
                       const glm::vec3& rotation)
                    { return entity.addComponent<atcg::TransformComponent>(position, scale, rotation); },
                    "positiona"_a,
                    "scale"_a,
                    "rotation"_a)
                .def(
                    "replaceTransformComponent",
                    [](atcg::Entity& entity, atcg::TransformComponent& transform)
                    { return entity.replaceComponent<atcg::TransformComponent>(transform); },
                    "transform"_a)
                .def(
                    "addGeometryComponent",
                    [](atcg::Entity& entity, const atcg::ref_ptr<atcg::Graph>& graph)
                    { return entity.addComponent<atcg::GeometryComponent>(graph); },
                    "graph"_a)
                .def(
                    "replaceGeometryComponent",
                    [](atcg::Entity& entity, atcg::GeometryComponent& geometry)
                    { return entity.replaceComponent<atcg::GeometryComponent>(geometry); },
                    "geometry"_a)
                .def(
                    "addMeshRenderComponent",
                    [](atcg::Entity& entity, const atcg::ref_ptr<atcg::Shader>& shader)
                    { return entity.addComponent<atcg::MeshRenderComponent>(shader); },
                    "shader"_a)
                .def(
                    "replaceMeshRenderComponent",
                    [](atcg::Entity& entity, atcg::MeshRenderComponent& component)
                    { return entity.replaceComponent<atcg::MeshRenderComponent>(component); },
                    "component"_a)
                .def(
                    "addPointRenderComponent",
                    [](atcg::Entity& entity,
                       const atcg::ref_ptr<atcg::Shader>& shader,
                       const glm::vec3& color,
                       float point_size)
                    { return entity.addComponent<atcg::PointRenderComponent>(shader, color, point_size); },
                    "shader"_a,
                    "color"_a,
                    "point_size"_a)
                .def(
                    "replacePointRenderComponent",
                    [](atcg::Entity& entity, atcg::PointRenderComponent& component)
                    { return entity.replaceComponent<atcg::PointRenderComponent>(component); },
                    "component_a")
                .def(
                    "addPointSphereRenderComponent",
                    [](atcg::Entity& entity, const atcg::ref_ptr<atcg::Shader>& shader, float point_size)
                    { return entity.addComponent<atcg::PointSphereRenderComponent>(shader, point_size); },
                    "shader"_a,
                    "point_size"_a)
                .def(
                    "replacePointSphereRenderComponent",
                    [](atcg::Entity& entity, atcg::PointSphereRenderComponent& component)
                    { return entity.replaceComponent<atcg::PointSphereRenderComponent>(component); },
                    "component"_a)
                .def(
                    "addEdgeRenderComponent",
                    [](atcg::Entity& entity, const glm::vec3& color)
                    { return entity.addComponent<atcg::EdgeRenderComponent>(color); },
                    "color"_a)
                .def(
                    "replaceEdgeRenderComponent",
                    [](atcg::Entity& entity, atcg::EdgeRenderComponent& component)
                    { return entity.replaceComponent<atcg::EdgeRenderComponent>(component); },
                    "component"_a)
                .def(
                    "addEdgeCylinderRenderComponent",
                    [](atcg::Entity& entity, float radius)
                    { return entity.addComponent<atcg::EdgeCylinderRenderComponent>(radius); },
                    "radius"_a)
                .def(
                    "replaceEdgeCylinderRenderComponent",
                    [](atcg::Entity& entity, atcg::EdgeCylinderRenderComponent& component)
                    { return entity.replaceComponent<atcg::EdgeCylinderRenderComponent>(component); },
                    "component"_a)
                .def("addInstanceRenderComponent",
                     [](atcg::Entity& entity) { return entity.addComponent<atcg::InstanceRenderComponent>(); })
                .def(
                    "replaceInstanceRenderComponent",
                    [](atcg::Entity& entity, atcg::InstanceRenderComponent& component)
                    { return entity.replaceComponent<atcg::InstanceRenderComponent>(component); },
                    "component"_a)
                .def(
                    "addPointLightComponent",
                    [](atcg::Entity& entity, float intensity, const glm::vec3& color)
                    { return entity.addComponent<atcg::PointLightComponent>(intensity, color); },
                    "intensity"_a,
                    "color"_a)
                .def(
                    "replacePointLightComponent",
                    [](atcg::Entity& entity, atcg::PointLightComponent& component)
                    { return entity.replaceComponent<atcg::PointLightComponent>(component); },
                    "component"_a)
                .def(
                    "addCameraComponent",
                    [](atcg::Entity& entity,
                       const atcg::ref_ptr<atcg::PerspectiveCamera>& camera,
                       const uint32_t width,
                       const uint32_t height)
                    { return entity.addComponent<atcg::CameraComponent>(camera, width, height); },
                    "camera"_a,
                    "width"_a,
                    "height"_a)
                .def(
                    "replaceCameraComponent",
                    [](atcg::Entity& entity, atcg::CameraComponent& component)
                    { return entity.replaceComponent<atcg::CameraComponent>(component); },
                    "component"_a)
                .def("addNameComponent",
                     [](atcg::Entity& entity, const std::string& name)
                     { return entity.addComponent<atcg::NameComponent>(name); })
                .def("replaceNameComponent",
                     [](atcg::Entity& entity, atcg::NameComponent& component)
                     { return entity.replaceComponent<atcg::NameComponent>(component); })
                .def("addScriptComponent",
                     [](atcg::Entity& entity, const atcg::ref_ptr<atcg::PythonScript>& script)
                     { return entity.addComponent<atcg::ScriptComponent>(script); })
                .def("replaceScriptComponent",
                     [](atcg::Entity& entity, atcg::ScriptComponent component)
                     { return entity.replaceComponent<atcg::ScriptComponent>(component); })
                .def("addMeshLightComponent",
                     [](atcg::Entity& entity) { return entity.addComponent<atcg::MeshLightComponent>(); })
                .def("replaceMeshLightComponent",
                     [](atcg::Entity& entity, atcg::MeshLightComponent& component)
                     { return entity.replaceComponent<atcg::MeshLightComponent>(component); })
                .def("hasTransformComponent", &atcg::Entity::hasComponent<atcg::TransformComponent>)
                .def("hasGeometryComponent", &atcg::Entity::hasComponent<atcg::GeometryComponent>)
                .def("hasMeshRenderComponent", &atcg::Entity::hasComponent<atcg::MeshRenderComponent>)
                .def("hasPointRenderComponent", &atcg::Entity::hasComponent<atcg::PointRenderComponent>)
                .def("hasPointSphereRenderComponent", &atcg::Entity::hasComponent<atcg::PointSphereRenderComponent>)
                .def("hasEdgeRenderComponent", &atcg::Entity::hasComponent<atcg::EdgeRenderComponent>)
                .def("hasEdgeCylinderRenderComponent", &atcg::Entity::hasComponent<atcg::EdgeCylinderRenderComponent>)
                .def("hasInstanceRenderComponent", &atcg::Entity::hasComponent<atcg::InstanceRenderComponent>)
                .def("hasPointLightComponent", &atcg::Entity::hasComponent<atcg::PointLightComponent>)
                .def("hasNameComponent", &atcg::Entity::hasComponent<atcg::NameComponent>)
                .def("hasScriptComponent", &atcg::Entity::hasComponent<atcg::ScriptComponent>)
                .def("hasMeshLightComponent", &atcg::Entity::hasComponent<atcg::MeshLightComponent>)
                .def("getTransformComponent", &atcg::Entity::getComponent<atcg::TransformComponent>)
                .def("getGeometryComponent", &atcg::Entity::getComponent<atcg::GeometryComponent>)
                .def("getMeshRenderComponent", &atcg::Entity::getComponent<atcg::MeshRenderComponent>)
                .def("getPointRenderComponent", &atcg::Entity::getComponent<atcg::PointRenderComponent>)
                .def("getPointSphereRenderComponent", &atcg::Entity::getComponent<atcg::PointSphereRenderComponent>)
                .def("getEdgeRenderComponent", &atcg::Entity::getComponent<atcg::EdgeRenderComponent>)
                .def("getPointLightComponent", &atcg::Entity::getComponent<atcg::PointLightComponent>)
                .def("getMeshLightComponent", &atcg::Entity::getComponent<atcg::MeshLightComponent>)
                .def("getEdgeCylinderRenderComponent", &atcg::Entity::getComponent<atcg::EdgeCylinderRenderComponent>)
                .def("getInstanceRenderComponent", &atcg::Entity::getComponent<atcg::InstanceRenderComponent>)
                .def("getScriptComponent", &atcg::Entity::getComponent<atcg::ScriptComponent>)
                .def("getCameraComponent", &atcg::Entity::getComponent<atcg::CameraComponent>)
                .def("getNameComponent", &atcg::Entity::getComponent<atcg::NameComponent>)
                .def("handle", &atcg::Entity::entity_handle);

            m_scene.def(py::init<>([]() { return atcg::make_ref<atcg::Scene>(); }))
                .def(
                    "createEntity",
                    [](const atcg::ref_ptr<atcg::Scene>& scene, const std::string& name)
                    { return scene->createEntity(name); },
                    "name"_a = "Entity")
                .def("getEntityByName", &atcg::Scene::getEntitiesByName, "name"_a)
                .def("getEntities",
                     [](const atcg::ref_ptr<atcg::Scene>& scene)
                     {
                         std::vector<atcg::Entity> entities;
                         auto view = scene->getAllEntitiesWith<atcg::IDComponent>();
                         for(auto e: view)
                         {
                             entities.push_back(atcg::Entity(e, scene.get()));
                         }
                         return entities;
                     })
                .def(
                    "getEntityByID",
                    [](const atcg::ref_ptr<atcg::Scene>& scene, uint64_t id)
                    {
                        atcg::UUID uuid(id);
                        return scene->getEntityByID(uuid);
                    },
                    "uuid"_a)
                .def(
                    "removeEntity",
                    [](const atcg::ref_ptr<atcg::Scene>& scene, uint64_t id)
                    {
                        atcg::UUID uuid(id);
                        scene->removeEntity(uuid);
                    },
                    "uuid"_a)
                .def(
                    "removeEntity",
                    [](const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity) { scene->removeEntity(entity); },
                    "entity"_a)
                .def("removeAllEntities", &atcg::Scene::removeAllEntites)
                .def("setCamera",
                     [](const atcg::ref_ptr<atcg::Scene>& self, const atcg::ref_ptr<atcg::PerspectiveCamera>& camera)
                     { self->setCamera(camera); })
                .def("getCamera", &atcg::Scene::getCamera)
                .def("removeCamera", &atcg::Scene::removeCamera)
                .def(
                    "setSkybox",
                    [](const atcg::ref_ptr<atcg::Scene>& scene, const atcg::ref_ptr<atcg::Image>& skybox)
                    { scene->setSkybox(skybox); },
                    "skybox"_a)
                .def(
                    "setSkybox",
                    [](const atcg::ref_ptr<atcg::Scene>& scene, const atcg::ref_ptr<atcg::Texture2D>& skybox)
                    { scene->setSkybox(skybox); },
                    "skybox"_a)
                .def("hasSkybox", &atcg::Scene::hasSkybox)
                .def("removeSkybox", &atcg::Scene::removeSkybox)
                .def("getSkyboxTexture", &atcg::Scene::getSkyboxTexture)
                .def("getSkyboxCubeMap", &atcg::Scene::getSkyboxCubemap)
                .def("getSelectedEntity", &atcg::Scene::getSelectedEntity)
                .def("selectEntity", &atcg::Scene::selectEntity, "entity"_a, "additive"_a = false)
                .def("deselectEntity", &atcg::Scene::deselectEntity, "entity"_a)
                .def("toggleEntitySelection", &atcg::Scene::toggleEntitySelection, "entity"_a)
                .def("clearSelection", &atcg::Scene::clearSelection)
                .def("isSelected", &atcg::Scene::isSelected, "entity"_a)
                .def("getSelectedEntities", &atcg::Scene::getSelectedEntities)
                .def("getActiveEntity", &atcg::Scene::getActiveEntity);

            m_scene_renderer
                .def("render",
                     [](const atcg::ref_ptr<atcg::Scene>& scene,
                        const atcg::ref_ptr<atcg::PerspectiveCamera>& camera,
                        const atcg::ref_ptr<atcg::Framebuffer>& framebuffer)
                     { atcg::SceneRenderer::render(scene, camera, framebuffer); })
                .def("setNumberMSAASamples", &atcg::SceneRenderer::setNumberMSAASamples, "samples"_a)
                .def("setShadowPassResolution", &atcg::SceneRenderer::setShadowPassResolution, "resolution"_a)
                .def("getNumberMSAASamples", &atcg::SceneRenderer::getNumberMSAASamples)
                .def("getShadowPassResolution", &atcg::SceneRenderer::getShadowPassResolution);

            m_hit_info.def_readonly("position", &atcg::Tracing::HitInfo::position)
                .def_readonly("incoming_direction", &atcg::Tracing::HitInfo::incoming_direction)
                .def_readonly("incoming_distance", &atcg::Tracing::HitInfo::incoming_distance)
                .def_readonly("normal", &atcg::Tracing::HitInfo::normal)
                .def_readonly("barys", &atcg::Tracing::HitInfo::barys)
                .def_readonly("uv", &atcg::Tracing::HitInfo::uv)
                .def_readonly("primitive_idx", &atcg::Tracing::HitInfo::primitive_idx)
                .def("isValid", &atcg::Tracing::HitInfo::isValid)
                .def("setInvalid", &atcg::Tracing::HitInfo::setInvalid)
                .def("isFinite", &atcg::Tracing::HitInfo::isFinite)
                .def("setInfinite", &atcg::Tracing::HitInfo::setInfinite);

            m.def("prepareAccelerationStructure", &atcg::Tracing::prepareAccelerationStructure);
            m.def("traceRay", atcg::Tracing::traceRay);
            m.def("traceRay",
                  [](atcg::Entity entity,
                     py::array_t<float> ray_origins,
                     py::array_t<float> ray_dirs,
                     float t_min,
                     float t_max)
                  {
                      py::buffer_info origin_info = ray_origins.request();
                      py::buffer_info dir_info    = ray_dirs.request();

                      auto result_hit      = py::array_t<bool>(origin_info.shape[0]);
                      bool* result_hit_ptr = (bool*)result_hit.request().ptr;

                      auto result_p           = py::array_t<float>(origin_info.size);
                      glm::vec3* result_p_ptr = (glm::vec3*)result_p.request().ptr;

                      auto result_idx          = py::array_t<uint32_t>(origin_info.shape[0]);
                      uint32_t* result_idx_ptr = (uint32_t*)result_idx.request().ptr;

                      uint32_t num_rays = origin_info.shape[0];

                      for(int i = 0; i < num_rays; ++i)
                      {
                          glm::vec3 o = *((glm::vec3*)origin_info.ptr + i);
                          glm::vec3 d = *((glm::vec3*)dir_info.ptr + i);

                          auto info = atcg::Tracing::traceRay(entity, o, d, t_min, t_max);

                          result_hit_ptr[i] = info.isValid();
                          result_p_ptr[i]   = info.position;
                          result_idx_ptr[i] = info.primitive_idx;
                      }

                      return std::make_tuple(result_hit, result_p, result_idx);
                  });

            m_clipboard.def("copyEntities", &atcg::Clipboard::copyEntities, "entities"_a)
                .def("pasteEntities", &atcg::Clipboard::pasteEntities, "scene"_a)
                .def("copyAsset", &atcg::Clipboard::copyAsset, "handle"_a)
                .def("pasteAsset", &atcg::Clipboard::pasteAsset)
                .def("hasEntities", &atcg::Clipboard::hasEntities)
                .def("hasAsset", &atcg::Clipboard::hasAsset)
                .def("clear", &atcg::Clipboard::clear);
        });
}
}    // namespace python
}    // namespace atcg
