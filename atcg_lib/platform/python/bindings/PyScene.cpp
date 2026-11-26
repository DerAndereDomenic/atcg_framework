#include "Bindings.h"

#ifdef ATCG_CUDA_BACKEND
    #include <BSDF/BSDF.h>
    #include <Emitter/Emitter.h>
    #include <Medium/Medium.h>
    #include <Medium/PhaseFunction.h>
    #include <Shape/Shape.h>
#endif

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
namespace
{
/**
 * @brief Bind has/get/replace/remove accessors of a component to the entity.
 * The getter returns a copy of the component.
 *
 * @tparam Component The component type
 * @param entity The entity class
 * @param name The name of the component in the accessor names, e.g. "TransformComponent"
 * @param replace_arg The argument name of the replace function
 */
template<typename Component>
void bindComponentAccessors(py::class_<atcg::Entity>& entity,
                            const std::string& name,
                            const char* replace_arg = "component")
{
    entity.def(("has" + name).c_str(), &atcg::Entity::hasComponent<Component>)
        .def(("get" + name).c_str(), &atcg::Entity::getComponent<Component>)
        .def(("replace" + name).c_str(),
             [](atcg::Entity& self, Component& component) { return self.replaceComponent<Component>(component); },
             py::arg(replace_arg))
        .def(("remove" + name).c_str(), &atcg::Entity::removeComponent<Component>);
}

/**
 * @brief Create a templated component revision for the component whose python type matches component_type.
 * A single entity creates a plain revision, several entities create a MultiEntityRevision.
 *
 * @tparam RevisionType The templated revision (ComponentAddedRevision, ComponentRemovedRevision, ...)
 * @tparam Component The component type to test
 * @tparam Rest The remaining component types
 * @param component_type The python type of the component
 * @param scene The scene
 * @param entities The entities
 *
 * @return The revision or nullptr if the component type is not supported
 */
template<template<typename> class RevisionType, typename Component, typename... Rest>
atcg::ref_ptr<atcg::Revision> createComponentRevision(const py::object& component_type,
                                                      const atcg::ref_ptr<atcg::Scene>& scene,
                                                      const std::vector<atcg::Entity>& entities)
{
    if(component_type.is(py::type::of<Component>()))
    {
        if(entities.size() == 1)
        {
            return atcg::make_ref<RevisionType<Component>>(scene, entities[0]);
        }
        return atcg::make_ref<atcg::MultiEntityRevision<RevisionType<Component>>>(scene, entities);
    }

    if constexpr(sizeof...(Rest) > 0)
    {
        return createComponentRevision<RevisionType, Rest...>(component_type, scene, entities);
    }
    else
    {
        ATCG_ERROR("RevisionStack: component type {0} does not support revisions",
                   py::str(component_type).cast<std::string>());
        return nullptr;
    }
}

/**
 * @brief Create a templated component revision for all components that are bound to python.
 *
 * @tparam RevisionType The templated revision (ComponentAddedRevision, ComponentRemovedRevision, ...)
 * @param component_type The python type of the component
 * @param scene The scene
 * @param entities The entities
 *
 * @return The revision or nullptr if the component type is not supported or no entity is given
 */
template<template<typename> class RevisionType>
atcg::ref_ptr<atcg::Revision> createComponentRevision(const py::object& component_type,
                                                      const atcg::ref_ptr<atcg::Scene>& scene,
                                                      const std::vector<atcg::Entity>& entities)
{
    if(!scene)
    {
        ATCG_ERROR("RevisionStack: scene is None");
        return nullptr;
    }

    if(entities.empty())
    {
        ATCG_ERROR("RevisionStack: no entities given");
        return nullptr;
    }

    return createComponentRevision<RevisionType,
                                   atcg::TransformComponent,
                                   atcg::GeometryComponent,
                                   atcg::MeshRenderComponent,
                                   atcg::PointRenderComponent,
                                   atcg::PointSphereRenderComponent,
                                   atcg::EdgeRenderComponent,
                                   atcg::EdgeCylinderRenderComponent,
                                   atcg::InstanceRenderComponent,
                                   atcg::CameraComponent,
                                   atcg::NameComponent,
                                   atcg::PointLightComponent,
                                   atcg::ScriptComponent,
                                   atcg::MeshLightComponent,
                                   atcg::TransparencyComponent,
                                   atcg::HomogeneousMediumComponent,
                                   atcg::HeterogeneousMediumComponent>(component_type, scene, entities);
}
}    // namespace

// Entities, scenes, scene renderer, ray tracing queries, clipboard and undo/redo
void bindScene(BindingContext& ctx)
{
    py::module_ m = ctx.module();

    auto m_entity = py::class_<atcg::Entity>(m, "Entity");
    auto m_scene  = py::class_<atcg::Scene, atcg::Asset, atcg::ref_ptr<atcg::Scene>>(m, "Scene");

    auto m_scene_renderer = m.def_submodule("SceneRenderer");

    auto m_entity_handle = py::class_<entt::entity>(m, "EntityHandle");

    auto m_hit_info = py::class_<atcg::Tracing::HitInfo>(m, "HitInfo");

    auto m_clipboard = m.def_submodule("Clipboard");

    auto m_revision_stack = m.def_submodule("RevisionStack");
    auto m_revision       = py::class_<atcg::Revision, atcg::ref_ptr<atcg::Revision>>(m, "Revision");
    auto m_composite_revision =
        py::class_<atcg::CompositeRevision, atcg::Revision, atcg::ref_ptr<atcg::CompositeRevision>>(m,
                                                                                                    "CompositeRevisio"
                                                                                                    "n");
    auto m_entity_added_revision =
        py::class_<atcg::EntityAddedRevision, atcg::Revision, atcg::ref_ptr<atcg::EntityAddedRevision>>(m,
                                                                                                        "EntityAddedRev"
                                                                                                        "ision");
    auto m_entity_removed_revision =
        py::class_<atcg::EntityRemovedRevision, atcg::Revision, atcg::ref_ptr<atcg::EntityRemovedRevision>>(m,
                                                                                                            "EntityRemo"
                                                                                                            "vedRevisio"
                                                                                                            "n");
    auto m_asset_added_revision =
        py::class_<atcg::AssetAddedRevision, atcg::Revision, atcg::ref_ptr<atcg::AssetAddedRevision>>(m,
                                                                                                      "AssetAddedRevisi"
                                                                                                      "on");
    auto m_asset_removed_revision =
        py::class_<atcg::AssetRemovedRevision, atcg::Revision, atcg::ref_ptr<atcg::AssetRemovedRevision>>(m,
                                                                                                          "AssetRemoved"
                                                                                                          "Revision");
    auto m_asset_edited_revision =
        py::class_<atcg::AssetEditedRevision, atcg::Revision, atcg::ref_ptr<atcg::AssetEditedRevision>>(m,
                                                                                                        "AssetEditedRev"
                                                                                                        "ision");

    ctx.define(
        [=]() mutable
        {
            m_entity_handle.def(py::init<uint32_t>(), "handle"_a)
                .def("__int__", [](entt::entity handle) { return (uint32_t)handle; });

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
                    "position"_a = glm::vec3(0),
                    "scale"_a    = glm::vec3(1),
                    "rotation"_a = glm::vec3(0))
                .def(
                    "addGeometryComponent",
                    [](atcg::Entity& entity, const atcg::ref_ptr<atcg::Graph>& graph)
                    { return entity.addComponent<atcg::GeometryComponent>(graph); },
                    "graph"_a)
                .def(
                    "addMeshRenderComponent",
                    [](atcg::Entity& entity, const atcg::ref_ptr<atcg::Shader>& shader)
                    {
                        if(!shader) return entity.addComponent<atcg::MeshRenderComponent>();
                        return entity.addComponent<atcg::MeshRenderComponent>(shader);
                    },
                    "shader"_a = nullptr)
                .def(
                    "addPointRenderComponent",
                    [](atcg::Entity& entity,
                       const atcg::ref_ptr<atcg::Shader>& shader,
                       const glm::vec3& color,
                       float point_size)
                    {
                        if(!shader) return entity.addComponent<atcg::PointRenderComponent>(color, point_size);
                        return entity.addComponent<atcg::PointRenderComponent>(shader, color, point_size);
                    },
                    "shader"_a     = nullptr,
                    "color"_a      = glm::vec3(1),
                    "point_size"_a = 1.0f)
                .def(
                    "addPointSphereRenderComponent",
                    [](atcg::Entity& entity, const atcg::ref_ptr<atcg::Shader>& shader, float point_size)
                    {
                        if(!shader) return entity.addComponent<atcg::PointSphereRenderComponent>(point_size);
                        return entity.addComponent<atcg::PointSphereRenderComponent>(shader, point_size);
                    },
                    "shader"_a     = nullptr,
                    "point_size"_a = 0.1f)
                .def(
                    "addEdgeRenderComponent",
                    [](atcg::Entity& entity, const glm::vec3& color)
                    { return entity.addComponent<atcg::EdgeRenderComponent>(color); },
                    "color"_a = glm::vec3(1))
                .def(
                    "addEdgeCylinderRenderComponent",
                    [](atcg::Entity& entity, float radius)
                    { return entity.addComponent<atcg::EdgeCylinderRenderComponent>(radius); },
                    "radius"_a = 0.001f)
                .def("addInstanceRenderComponent",
                     [](atcg::Entity& entity) { return entity.addComponent<atcg::InstanceRenderComponent>(); })
                .def(
                    "addPointLightComponent",
                    [](atcg::Entity& entity, float intensity, const glm::vec3& color)
                    { return entity.addComponent<atcg::PointLightComponent>(intensity, color); },
                    "intensity"_a = 1.0f,
                    "color"_a     = glm::vec3(1))
                .def(
                    "addCameraComponent",
                    [](atcg::Entity& entity,
                       const atcg::ref_ptr<atcg::Camera>& camera,
                       const uint32_t width,
                       const uint32_t height)
                    { return entity.addComponent<atcg::CameraComponent>(camera, width, height); },
                    "camera"_a,
                    "width"_a  = 1024,
                    "height"_a = 1024)
                .def(
                    "addNameComponent",
                    [](atcg::Entity& entity, const std::string& name)
                    { return entity.addOrReplaceComponent<atcg::NameComponent>(name); },
                    "name"_a)
                .def(
                    "addScriptComponent",
                    [](atcg::Entity& entity, const atcg::ref_ptr<atcg::Script>& script)
                    {
                        if(!script) return entity.addComponent<atcg::ScriptComponent>();
                        return entity.addComponent<atcg::ScriptComponent>(script);
                    },
                    "script"_a)
                .def("addMeshLightComponent",
                     [](atcg::Entity& entity) { return entity.addComponent<atcg::MeshLightComponent>(); })
                .def(
                    "addTransparencyComponent",
                    [](atcg::Entity& entity, bool transparent)
                    { return entity.addComponent<atcg::TransparencyComponent>(transparent); },
                    "transparent"_a = true)
                .def("addHomogeneousMediumComponent",
                     [](atcg::Entity& entity) { return entity.addComponent<atcg::HomogeneousMediumComponent>(); })
                .def(
                    "addHomogeneousMediumComponent",
                    [](atcg::Entity& entity, const atcg::HomogeneousMediumComponent& component)
                    { return entity.addComponent<atcg::HomogeneousMediumComponent>(component); },
                    "component"_a)
                .def("addHeterogeneousMediumComponent",
                     [](atcg::Entity& entity) { return entity.addComponent<atcg::HeterogeneousMediumComponent>(); })
                .def(
                    "addHeterogeneousMediumComponent",
                    [](atcg::Entity& entity, const atcg::HeterogeneousMediumComponent& component)
                    { return entity.addComponent<atcg::HeterogeneousMediumComponent>(component); },
                    "component"_a)
                .def("hasIDComponent", &atcg::Entity::hasComponent<atcg::IDComponent>)
                .def("getIDComponent", &atcg::Entity::getComponent<atcg::IDComponent>)
                .def("getID",
                     [](atcg::Entity& entity) -> uint64_t
                     {
                         if(!entity.hasComponent<atcg::IDComponent>())
                         {
                             ATCG_WARN("Entity.getID: entity has no IDComponent");
                             return 0;
                         }
                         return (uint64_t)entity.getComponent<atcg::IDComponent>().ID();
                     })
                .def("handle", &atcg::Entity::entity_handle)
                .def("scene",
                     [](const atcg::Entity& entity) -> atcg::ref_ptr<atcg::Scene>
                     {
                         atcg::Scene* scene = entity.scene();
                         if(!scene) return nullptr;
                         return scene->weak_from_this().lock();
                     })
                .def("isValid", [](const atcg::Entity& entity) { return entity.scene() != nullptr && (bool)entity; })
                .def("__bool__", [](const atcg::Entity& entity) { return entity.scene() != nullptr && (bool)entity; })
                .def(
                    "__eq__",
                    [](const atcg::Entity& self, const atcg::Entity& other) { return self == other; },
                    py::is_operator())
                .def("__hash__",
                     [](const atcg::Entity& self)
                     { return py::hash(py::make_tuple(self.entity_handle(), (uintptr_t)self.scene())); });

            bindComponentAccessors<atcg::TransformComponent>(m_entity, "TransformComponent", "transform");
            bindComponentAccessors<atcg::GeometryComponent>(m_entity, "GeometryComponent", "geometry");
            bindComponentAccessors<atcg::MeshRenderComponent>(m_entity, "MeshRenderComponent");
            bindComponentAccessors<atcg::PointRenderComponent>(m_entity, "PointRenderComponent");
            bindComponentAccessors<atcg::PointSphereRenderComponent>(m_entity, "PointSphereRenderComponent");
            bindComponentAccessors<atcg::EdgeRenderComponent>(m_entity, "EdgeRenderComponent");
            bindComponentAccessors<atcg::EdgeCylinderRenderComponent>(m_entity, "EdgeCylinderRenderComponent");
            bindComponentAccessors<atcg::InstanceRenderComponent>(m_entity, "InstanceRenderComponent");
            bindComponentAccessors<atcg::PointLightComponent>(m_entity, "PointLightComponent");
            bindComponentAccessors<atcg::CameraComponent>(m_entity, "CameraComponent");
            bindComponentAccessors<atcg::NameComponent>(m_entity, "NameComponent");
            bindComponentAccessors<atcg::ScriptComponent>(m_entity, "ScriptComponent");
            bindComponentAccessors<atcg::MeshLightComponent>(m_entity, "MeshLightComponent");
            bindComponentAccessors<atcg::TransparencyComponent>(m_entity, "TransparencyComponent");
            bindComponentAccessors<atcg::HomogeneousMediumComponent>(m_entity, "HomogeneousMediumComponent");
            bindComponentAccessors<atcg::HeterogeneousMediumComponent>(m_entity, "HeterogeneousMediumComponent");

#ifdef ATCG_CUDA_BACKEND
            // Components of the entities of an OptixScene. The python types are declared in bindOptix.
            m_entity.def("hasBSDFComponent", &atcg::Entity::hasComponent<atcg::BSDFComponent>)
                .def("getBSDFComponent", &atcg::Entity::getComponent<atcg::BSDFComponent>)
                .def("hasEmitterComponent", &atcg::Entity::hasComponent<atcg::EmitterComponent>)
                .def("getEmitterComponent", &atcg::Entity::getComponent<atcg::EmitterComponent>)
                .def("hasMediumComponent", &atcg::Entity::hasComponent<atcg::MediumComponent>)
                .def("getMediumComponent", &atcg::Entity::getComponent<atcg::MediumComponent>)
                .def("hasPhaseFunctionComponent", &atcg::Entity::hasComponent<atcg::PhaseFunctionComponent>)
                .def("getPhaseFunctionComponent", &atcg::Entity::getComponent<atcg::PhaseFunctionComponent>)
                .def("hasShapeComponent", &atcg::Entity::hasComponent<atcg::ShapeComponent>)
                .def("getShapeComponent", &atcg::Entity::getComponent<atcg::ShapeComponent>);
#endif

            m_scene.def(py::init<>([]() { return atcg::make_ref<atcg::Scene>(); }))
                .def(
                    "createEntity",
                    [](const atcg::ref_ptr<atcg::Scene>& scene, const std::string& name)
                    { return scene->createEntity(name); },
                    "name"_a = "Entity")
                .def(
                    "createEntity",
                    [](const atcg::ref_ptr<atcg::Scene>& scene,
                       entt::entity handle,
                       uint64_t uuid,
                       const std::string& name) { return scene->createEntity(handle, atcg::UUID(uuid), name); },
                    "handle"_a,
                    "uuid"_a,
                    "name"_a)
                .def("getEntitiesByName", &atcg::Scene::getEntitiesByName, "name"_a)
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
                .def("setCamera", &atcg::Scene::setCamera, "camera"_a)
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
                .def("getSkybox", &atcg::Scene::getSkybox)
                .def("getSkyboxTexture", &atcg::Scene::getSkyboxTexture)
                .def("getSkyboxCubeMap", &atcg::Scene::getSkyboxCubemap)
                .def("getSelectedEntity", &atcg::Scene::getSelectedEntity)
                .def("selectEntity", &atcg::Scene::selectEntity, "entity"_a, "additive"_a = false)
                .def("deselectEntity", &atcg::Scene::deselectEntity, "entity"_a)
                .def("toggleEntitySelection", &atcg::Scene::toggleEntitySelection, "entity"_a)
                .def("clearSelection", &atcg::Scene::clearSelection)
                .def("isSelected", &atcg::Scene::isSelected, "entity"_a)
                .def("getSelectedEntities", &atcg::Scene::getSelectedEntities)
                .def("getActiveEntity", &atcg::Scene::getActiveEntity)
                .def_static("getStaticType", &atcg::Scene::getStaticType);

            m_scene_renderer
                .def(
                    "render",
                    [](const atcg::ref_ptr<atcg::Scene>& scene,
                       const atcg::ref_ptr<atcg::Camera>& camera,
                       const atcg::ref_ptr<atcg::Framebuffer>& framebuffer,
                       bool draw_cameras) { atcg::SceneRenderer::render(scene, camera, framebuffer, draw_cameras); },
                    "scene"_a,
                    "camera"_a,
                    "framebuffer"_a,
                    "draw_cameras"_a = true)
                .def("getRenderGraph", []() { return atcg::SceneRenderer::getRenderGraph(); })
                .def(
                    "setRenderGraph",
                    [](atcg::ref_ptr<atcg::RenderGraph> graph) { atcg::SceneRenderer::setRenderGraph(graph); },
                    "graph"_a)
                .def("setNumberMSAASamples", &atcg::SceneRenderer::setNumberMSAASamples, "samples"_a)
                .def("setShadowPassResolution", &atcg::SceneRenderer::setShadowPassResolution, "resolution"_a)
                .def("getNumberMSAASamples", &atcg::SceneRenderer::getNumberMSAASamples)
                .def("getShadowPassResolution", &atcg::SceneRenderer::getShadowPassResolution);

            m_hit_info.def(py::init<>())
                .def_readonly("position", &atcg::Tracing::HitInfo::position)
                .def_readonly("incoming_direction", &atcg::Tracing::HitInfo::incoming_direction)
                .def_readonly("incoming_distance", &atcg::Tracing::HitInfo::incoming_distance)
                .def_readonly("pdf", &atcg::Tracing::HitInfo::pdf)
                .def_readonly("normal", &atcg::Tracing::HitInfo::normal)
                .def_readonly("barys", &atcg::Tracing::HitInfo::barys)
                .def_readonly("uv", &atcg::Tracing::HitInfo::uv)
                .def_readonly("primitive_idx", &atcg::Tracing::HitInfo::primitive_idx)
                .def_readonly("entity_id", &atcg::Tracing::HitInfo::entity_id)
                .def_readonly("dx_du", &atcg::Tracing::HitInfo::dx_du)
                .def_readonly("dx_dv", &atcg::Tracing::HitInfo::dx_dv)
                .def("isValid", &atcg::Tracing::HitInfo::isValid)
                .def("setInvalid", &atcg::Tracing::HitInfo::setInvalid)
                .def("isFinite", &atcg::Tracing::HitInfo::isFinite)
                .def("setInfinite", &atcg::Tracing::HitInfo::setInfinite);

            m.def("prepareAccelerationStructure", &atcg::Tracing::prepareAccelerationStructure, "entity"_a);
            m.def("traceRay", &atcg::Tracing::traceRay, "entity"_a, "ray_origin"_a, "ray_dir"_a, "t_min"_a, "t_max"_a);
            m.def(
                "traceRay",
                [](atcg::Entity entity,
                   py::array_t<float, py::array::c_style | py::array::forcecast> ray_origins,
                   py::array_t<float, py::array::c_style | py::array::forcecast> ray_dirs,
                   float t_min,
                   float t_max)
                {
                    py::buffer_info origin_info = ray_origins.request();
                    py::buffer_info dir_info    = ray_dirs.request();

                    if(origin_info.ndim != 2 || origin_info.shape[1] != 3 || dir_info.ndim != 2 ||
                       dir_info.shape[1] != 3 || origin_info.shape[0] != dir_info.shape[0])
                    {
                        ATCG_ERROR("traceRay: ray_origins and ray_dirs must both have the shape (N, 3)");
                        return std::make_tuple(py::array_t<bool>(py::ssize_t {0}),
                                               py::array_t<float>(std::vector<py::ssize_t> {0, 3}),
                                               py::array_t<uint32_t>(py::ssize_t {0}));
                    }

                    py::ssize_t num_rays = origin_info.shape[0];

                    auto result_hit      = py::array_t<bool>(num_rays);
                    bool* result_hit_ptr = (bool*)result_hit.request().ptr;

                    auto result_p           = py::array_t<float>(std::vector<py::ssize_t> {num_rays, 3});
                    glm::vec3* result_p_ptr = (glm::vec3*)result_p.request().ptr;

                    auto result_idx          = py::array_t<uint32_t>(num_rays);
                    uint32_t* result_idx_ptr = (uint32_t*)result_idx.request().ptr;

                    const glm::vec3* origins    = (const glm::vec3*)origin_info.ptr;
                    const glm::vec3* directions = (const glm::vec3*)dir_info.ptr;

                    for(py::ssize_t i = 0; i < num_rays; ++i)
                    {
                        auto info = atcg::Tracing::traceRay(entity, origins[i], directions[i], t_min, t_max);

                        result_hit_ptr[i] = info.isValid();
                        result_p_ptr[i]   = info.position;
                        result_idx_ptr[i] = info.primitive_idx;
                    }

                    return std::make_tuple(result_hit, result_p, result_idx);
                },
                "entity"_a,
                "ray_origins"_a,
                "ray_dirs"_a,
                "t_min"_a,
                "t_max"_a);

            m_clipboard.def("copyEntities", &atcg::Clipboard::copyEntities, "entities"_a)
                .def("pasteEntities", &atcg::Clipboard::pasteEntities, "scene"_a)
                .def("copyAsset", &atcg::Clipboard::copyAsset, "handle"_a)
                .def("pasteAsset", &atcg::Clipboard::pasteAsset)
                .def("hasEntities", &atcg::Clipboard::hasEntities)
                .def("hasAsset", &atcg::Clipboard::hasAsset)
                .def("clear", &atcg::Clipboard::clear);

            // Revisions
            m_revision.def("apply", &atcg::Revision::apply)
                .def("rollback", &atcg::Revision::rollback)
                .def("record_start_state", &atcg::Revision::record_start_state)
                .def("record_end_state", &atcg::Revision::record_end_state);

            m_composite_revision.def(py::init<>())
                .def(
                    "add",
                    [](atcg::CompositeRevision& self, const atcg::ref_ptr<atcg::Revision>& revision)
                    {
                        if(!revision)
                        {
                            ATCG_WARN("CompositeRevision.add: revision is None");
                            return;
                        }
                        self.add(revision);
                    },
                    "revision"_a)
                .def("empty", &atcg::CompositeRevision::empty);

            m_entity_added_revision.def(py::init<const atcg::ref_ptr<atcg::Scene>&, atcg::Entity>(),
                                        "scene"_a,
                                        "entity"_a);
            m_entity_removed_revision.def(py::init<const atcg::ref_ptr<atcg::Scene>&, atcg::Entity>(),
                                          "scene"_a,
                                          "entity"_a);
            m_asset_added_revision.def(py::init<atcg::AssetHandle>(), "handle"_a);
            m_asset_removed_revision.def(py::init<atcg::AssetHandle>(), "handle"_a);
            m_asset_edited_revision.def(py::init<atcg::AssetHandle>(), "handle"_a);

            // Revision stack
            m_revision_stack
                .def(
                    "startRecording",
                    [](const atcg::ref_ptr<atcg::Revision>& revision)
                    {
                        if(!revision)
                        {
                            ATCG_WARN("RevisionStack.startRecording: revision is None");
                            return;
                        }
                        if(atcg::RevisionStack::isRecording())
                        {
                            ATCG_WARN("RevisionStack.startRecording: a revision is already being recorded");
                            return;
                        }
                        atcg::RevisionStack::startRecording(revision);
                    },
                    "revision"_a)
                .def("endRecording",
                     []()
                     {
                         if(!atcg::RevisionStack::isRecording())
                         {
                             ATCG_WARN("RevisionStack.endRecording: no revision is being recorded");
                             return;
                         }
                         atcg::RevisionStack::endRecording();
                     })
                .def("apply", &atcg::RevisionStack::apply)
                .def("rollback", &atcg::RevisionStack::rollback)
                .def("numUndos", &atcg::RevisionStack::numUndos)
                .def("numRedos", &atcg::RevisionStack::numRedos)
                .def("isRecording", &atcg::RevisionStack::isRecording)
                .def("totalRevisions", &atcg::RevisionStack::totalRevisions)
                .def("clearCache",
                     []()
                     {
                         if(atcg::RevisionStack::isRecording())
                         {
                             ATCG_WARN("RevisionStack.clearCache: can't clear the cache while recording");
                             return;
                         }
                         atcg::RevisionStack::clearChache();
                     })
                .def(
                    "createComponentAddedRevision",
                    [](const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity, const py::object& component_type)
                    { return createComponentRevision<atcg::ComponentAddedRevision>(component_type, scene, {entity}); },
                    "scene"_a,
                    "entity"_a,
                    "component_type"_a)
                .def(
                    "createComponentAddedRevision",
                    [](const atcg::ref_ptr<atcg::Scene>& scene,
                       const std::vector<atcg::Entity>& entities,
                       const py::object& component_type)
                    { return createComponentRevision<atcg::ComponentAddedRevision>(component_type, scene, entities); },
                    "scene"_a,
                    "entities"_a,
                    "component_type"_a)
                .def(
                    "createComponentRemovedRevision",
                    [](const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity, const py::object& component_type) {
                        return createComponentRevision<atcg::ComponentRemovedRevision>(component_type, scene, {entity});
                    },
                    "scene"_a,
                    "entity"_a,
                    "component_type"_a)
                .def(
                    "createComponentRemovedRevision",
                    [](const atcg::ref_ptr<atcg::Scene>& scene,
                       const std::vector<atcg::Entity>& entities,
                       const py::object& component_type) {
                        return createComponentRevision<atcg::ComponentRemovedRevision>(component_type, scene, entities);
                    },
                    "scene"_a,
                    "entities"_a,
                    "component_type"_a)
                .def(
                    "createComponentEditedRevision",
                    [](const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity, const py::object& component_type)
                    { return createComponentRevision<atcg::ComponentEditedRevision>(component_type, scene, {entity}); },
                    "scene"_a,
                    "entity"_a,
                    "component_type"_a)
                .def(
                    "createComponentEditedRevision",
                    [](const atcg::ref_ptr<atcg::Scene>& scene,
                       const std::vector<atcg::Entity>& entities,
                       const py::object& component_type)
                    { return createComponentRevision<atcg::ComponentEditedRevision>(component_type, scene, entities); },
                    "scene"_a,
                    "entities"_a,
                    "component_type"_a);
        });
}
}    // namespace python
}    // namespace atcg
