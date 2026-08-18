#include <Scene/SceneAdapter.h>
#include <Scene/Components.h>
#include <Emitter/PointEmitter.h>
#include <Emitter/MeshEmitter.h>
#include <Shape/MeshShape.h>
#include <Core/Path.h>
#include <Core/Assert.h>
#include <Material/MaterialRegistry.h>

// !TEST
#include <Film/HDRFilm.h>
#include <Sensor/PinholeCamera.h>

namespace atcg
{

//! temp
struct BSDFComponent
{
    BSDFComponent(AssetHandle bsdf_handle) : bsdf_handle(bsdf_handle) {}
    AssetHandle bsdf_handle = 0;
};

struct EmitterComponent
{
    EmitterComponent(const atcg::ref_ptr<Emitter>& emitter) : emitter(emitter) {}
    atcg::ref_ptr<Emitter> emitter;
};

struct ShapeComponent
{
    ShapeComponent(const atcg::ref_ptr<Shape>& shape) : shape(shape) {}
    atcg::ref_ptr<Shape> shape;
};

template<typename T>
void SceneAdapter::prepareComponent(const atcg::ref_ptr<OptixScene>& result, Entity entity)
{
}

template<>
void SceneAdapter::prepareComponent<MeshRenderComponent>(const atcg::ref_ptr<OptixScene>& result, Entity entity)
{
    if(!entity.hasComponent<MeshRenderComponent>()) return;
    if(!entity.hasComponent<GeometryComponent>()) return;

    auto original_name = entity.getComponent<NameComponent>().name();
    auto new_entity    = result->createEntity(original_name);

    MeshRenderComponent& component = entity.getComponent<MeshRenderComponent>();

    if(!component.visible) return;

    auto& transform = entity.getComponent<TransformComponent>();

    auto geometry = entity.getComponent<GeometryComponent>();
    auto shape_it = _shape_cache.find(geometry.graph_handle);
    if(shape_it == _shape_cache.end()) return;
    auto shape = shape_it->second;

    auto bsdf_it = _bsdf_cache.find(component.material_handle);
    if(bsdf_it == _bsdf_cache.end()) return;
    auto bsdf = bsdf_it->second;

    if(entity.hasComponent<MeshLightComponent>())
    {
        auto& mesh_light_component = entity.getComponent<MeshLightComponent>();
        Dictionary emitter_data;
        emitter_data.setValue<atcg::ref_ptr<MeshShape>>("shape", std::dynamic_pointer_cast<MeshShape>(shape));
        emitter_data.setValue("transform", transform.getModel());
        emitter_data.setValue("emission_scaling", mesh_light_component.intensity);
        emitter_data.setValue("texture_emissive", mesh_light_component.getEmissiveTexture());

        atcg::ref_ptr<MeshEmitter> mesh_emitter = atcg::make_ref<MeshEmitter>(emitter_data);
        mesh_emitter->initializePipeline(_pipeline, _sbt);

        new_entity.addComponent<EmitterComponent>(mesh_emitter);
    }

    // Dictionary shape_data;

    if(entity.hasComponent<MediumComponent>())
    {
        auto& component = entity.getComponent<MediumComponent>();

        if(component.medium() && component.phase_function())
        {
            component.medium()->setPhaseFunction(component.phase_function());
            component.phase_function()->initializePipeline(_pipeline, _sbt);
            component.medium()->initializePipeline(_pipeline, _sbt);

            new_entity.addComponent<MediumComponent>(component.medium());
        }
        // new_entity.addComponent<PhaseFunctionComponent>(phase);
    }

    new_entity.addComponent<ShapeComponent>(shape);
    new_entity.addComponent<BSDFComponent>(component.material_handle);
    new_entity.addComponent<TransformComponent>(transform);
    new_entity.addComponent<int32_t>((int32_t)entity.entity_handle());

    // Add AABB of the shape to the scene AABB
    atcg::BoundingBox shape_aabb = geometry.graph()->getBoundingBox();
    shape_aabb                   = atcg::Utils::transformBoundingBox(shape_aabb, transform.getModel());
    _scene_aabb                  = _scene_aabb + shape_aabb;
}

template<>
void SceneAdapter::prepareComponent<PointSphereRenderComponent>(const atcg::ref_ptr<OptixScene>& result, Entity entity)
{
    if(!entity.hasComponent<PointSphereRenderComponent>()) return;
    if(!entity.hasComponent<GeometryComponent>()) return;

    PointSphereRenderComponent& component = entity.getComponent<PointSphereRenderComponent>();

    if(!component.visible) return;

    auto& transform            = entity.getComponent<TransformComponent>();
    glm::mat4 global_transform = transform.getModel();
    auto material              = component.material();

    auto graph = atcg::IO::read_mesh((atcg::resource_directory() / "sphere_low.obj").string());
    atcg::Dictionary shape_dict;
    shape_dict.setValue("mesh", graph);
    atcg::ref_ptr<MeshShape> shape = atcg::make_ref<MeshShape>(shape_dict);
    shape->initializePipeline(_pipeline, _sbt);
    shape->prepareAccelerationStructure(_context);

    auto bsdf_it = _bsdf_cache.find(component.material_handle);
    if(bsdf_it == _bsdf_cache.end()) return;
    auto bsdf = bsdf_it->second;

    auto mesh = entity.getComponent<GeometryComponent>().graph();
    if(!mesh) return;
    uint32_t n_instances = mesh->n_vertices();

    torch::Tensor offsets = mesh->getHostPositions();
    torch::Tensor colors  = mesh->getHostColors();

    // Logic from base.vs
    glm::vec3 scale_model =
        glm::vec3(glm::length(global_transform[0]), glm::length(global_transform[1]), glm::length(global_transform[2]));
    glm::vec3 scale_point     = glm::vec3(component.point_size);
    glm::mat4 inv_scale_model = glm::mat4(1);
    inv_scale_model[0][0]     = 1.0 / scale_model.x;
    inv_scale_model[1][1]     = 1.0 / scale_model.y;
    inv_scale_model[2][2]     = 1.0 / scale_model.z;

    glm::mat4 scale_primitive = glm::mat4(1);
    scale_primitive[0][0]     = scale_point.x;
    scale_primitive[1][1]     = scale_point.y;
    scale_primitive[2][2]     = scale_point.z;

    auto original_name = entity.getComponent<NameComponent>().name();
    for(int i = 0; i < n_instances; ++i)
    {
        auto new_entity = result->createEntity(original_name);

        glm::vec3 offset =
            glm::vec3(offsets[i][0].item<float>(), offsets[i][1].item<float>(), offsets[i][2].item<float>());
        glm::vec3 color = glm::vec3(colors[i][0].item<float>(), colors[i][1].item<float>(), colors[i][2].item<float>());
        glm::mat4 total_transform = glm::translate(glm::vec3(global_transform * glm::vec4(offset, 0))) *
                                    global_transform * inv_scale_model * scale_primitive;

        new_entity.addComponent<ShapeComponent>(shape);
        new_entity.addComponent<BSDFComponent>(component.material_handle);
        new_entity.addComponent<TransformComponent>(total_transform);
        new_entity.addComponent<int32_t>((int32_t)entity.entity_handle());
        new_entity.addComponent<glm::vec3>(color);

        // Add AABB of the shape to the scene AABB
        atcg::BoundingBox shape_aabb = graph->getBoundingBox();
        shape_aabb                   = atcg::Utils::transformBoundingBox(shape_aabb, transform.getModel());
        _scene_aabb                  = _scene_aabb + shape_aabb;
    }

    mesh->unmapAllHostPointers();
}

template<>
void SceneAdapter::prepareComponent<EdgeCylinderRenderComponent>(const atcg::ref_ptr<OptixScene>& result, Entity entity)
{
    if(!entity.hasComponent<EdgeCylinderRenderComponent>()) return;
    if(!entity.hasComponent<GeometryComponent>()) return;

    EdgeCylinderRenderComponent& component = entity.getComponent<EdgeCylinderRenderComponent>();

    if(!component.visible) return;

    auto& transform            = entity.getComponent<TransformComponent>();
    glm::mat4 global_transform = transform.getModel();
    auto material              = component.material();

    auto graph = atcg::IO::read_mesh((atcg::resource_directory() / "cylinder.obj").string());
    atcg::Dictionary shape_dict;
    shape_dict.setValue("mesh", graph);
    atcg::ref_ptr<MeshShape> shape = atcg::make_ref<MeshShape>(shape_dict);
    shape->initializePipeline(_pipeline, _sbt);
    shape->prepareAccelerationStructure(_context);

    auto bsdf_it = _bsdf_cache.find(component.material_handle);
    if(bsdf_it == _bsdf_cache.end()) return;
    auto bsdf = bsdf_it->second;

    auto mesh = entity.getComponent<GeometryComponent>().graph();
    if(!mesh) return;
    uint32_t n_instances = mesh->n_edges();

    torch::Tensor positions = mesh->getHostPositions();
    torch::Tensor indices   = mesh->getHostEdges();

    auto original_name = entity.getComponent<NameComponent>().name();
    // Logic from cylinder_edge.vs
    for(int i = 0; i < n_instances; ++i)
    {
        auto new_entity = result->createEntity(original_name);

        int edge_x = int(indices[i][0].item<float>());
        int edge_y = int(indices[i][1].item<float>());
        glm::vec3 edge_color =
            glm::vec3(indices[i][2].item<float>(), indices[i][3].item<float>(), indices[i][4].item<float>());
        float edge_radius        = indices[i][5].item<float>();
        glm::vec3 aInstanceStart = glm::vec3(global_transform * glm::vec4(positions[edge_x][0].item<float>(),
                                                                          positions[edge_x][1].item<float>(),
                                                                          positions[edge_x][2].item<float>(),
                                                                          1));

        glm::vec3 aInstanceEnd = glm::vec3(global_transform * glm::vec4(positions[edge_y][0].item<float>(),
                                                                        positions[edge_y][1].item<float>(),
                                                                        positions[edge_y][2].item<float>(),
                                                                        1));

        glm::vec3 axis         = (aInstanceEnd - aInstanceStart);
        glm::vec3 middle_point = aInstanceStart + axis / 2.0f;

        glm::mat4 model_scale = glm::mat4(edge_radius * component.radius);
        model_scale[1].y      = length(axis) / 2.0;
        model_scale[3].w      = 1;

        glm::mat4 model_translate = glm::mat4(1);
        model_translate[3]        = glm::vec4(middle_point, 1);

        axis        = glm::normalize(axis);
        glm::vec3 x = glm::normalize(glm::cross(glm::vec3(0, axis.z, 1.0f - axis.z), axis));
        glm::vec3 z = glm::normalize(glm::cross(x, axis));

        glm::mat4 model_rotation =
            glm::mat4(glm::vec4(x, 0), glm::vec4(axis, 0), glm::vec4(z, 0), glm::vec4(0, 0, 0, 1));

        glm::mat4 model_edge = model_translate * model_rotation * model_scale;

        new_entity.addComponent<ShapeComponent>(shape);
        new_entity.addComponent<BSDFComponent>(component.material_handle);
        new_entity.addComponent<TransformComponent>(model_edge);
        new_entity.addComponent<int32_t>((int32_t)entity.entity_handle());
        new_entity.addComponent<glm::vec3>(edge_color);

        // Add AABB of the shape to the scene AABB
        atcg::BoundingBox shape_aabb = graph->getBoundingBox();
        shape_aabb                   = atcg::Utils::transformBoundingBox(shape_aabb, transform.getModel());
        _scene_aabb                  = _scene_aabb + shape_aabb;
    }

    mesh->unmapAllHostPointers();
}

template<>
void SceneAdapter::prepareComponent<InstanceRenderComponent>(const atcg::ref_ptr<OptixScene>& result, Entity entity)
{
    if(!entity.hasComponent<InstanceRenderComponent>()) return;
    if(!entity.hasComponent<GeometryComponent>()) return;

    InstanceRenderComponent& component = entity.getComponent<InstanceRenderComponent>();

    if(!component.visible) return;

    if(component.instance_vbos.size() != 2)
    {
        ATCG_WARN("Expecting size two for number of instances in path tracer...");
        return;
    }

    auto& transform            = entity.getComponent<TransformComponent>();
    glm::mat4 global_transform = transform.getModel();
    auto material              = component.material();

    auto geometry = entity.getComponent<GeometryComponent>();
    auto shape_it = _shape_cache.find(geometry.graph_handle);
    if(shape_it == _shape_cache.end()) return;

    auto shape = shape_it->second;

    auto bsdf_it = _bsdf_cache.find(component.material_handle);
    if(bsdf_it == _bsdf_cache.end()) return;
    auto bsdf = bsdf_it->second;

    auto transform_vbo   = component.instance_vbos[0];
    auto color_vbo       = component.instance_vbos[1];
    uint32_t n_instances = transform_vbo->size() / transform_vbo->getLayout().getStride();

    ATCG_ASSERT(n_instances == (color_vbo->size() / color_vbo->getLayout().getStride()),
                "Instance buffers have wrong size");

    glm::mat4* transforms = transform_vbo->getHostPointer<glm::mat4>();
    glm::vec4* colors     = color_vbo->getHostPointer<glm::vec4>();

    auto original_name = entity.getComponent<NameComponent>().name();
    for(int i = 0; i < n_instances; ++i)
    {
        auto new_entity = result->createEntity(original_name);
        new_entity.addComponent<ShapeComponent>(shape);
        new_entity.addComponent<BSDFComponent>(component.material_handle);
        new_entity.addComponent<TransformComponent>(global_transform * transforms[i]);
        new_entity.addComponent<int32_t>((int32_t)entity.entity_handle());
        new_entity.addComponent<glm::vec3>(colors[i]);

        // Add AABB of the shape to the scene AABB
        atcg::BoundingBox shape_aabb = geometry.graph()->getBoundingBox();
        shape_aabb                   = atcg::Utils::transformBoundingBox(shape_aabb, transform.getModel());
        _scene_aabb                  = _scene_aabb + shape_aabb;
    }

    transform_vbo->unmapHostPointers();
    color_vbo->unmapHostPointers();
}

template<>
void SceneAdapter::prepareComponent<MeshLightComponent>(const atcg::ref_ptr<OptixScene>& result, Entity entity)
{
    if(!entity.hasComponent<MeshLightComponent>()) return;
    if(entity.hasComponent<MeshRenderComponent>()) return;    // Light source already added here
    if(!entity.hasComponent<GeometryComponent>()) return;

    MeshLightComponent& component = entity.getComponent<MeshLightComponent>();

    auto& transform = entity.getComponent<TransformComponent>();

    auto geometry = entity.getComponent<GeometryComponent>();
    auto shape_it = _shape_cache.find(geometry.graph_handle);
    if(shape_it == _shape_cache.end()) return;

    auto shape = shape_it->second;

    auto& mesh_light_component = entity.getComponent<MeshLightComponent>();
    Dictionary emitter_data;
    emitter_data.setValue<atcg::ref_ptr<MeshShape>>("shape", std::dynamic_pointer_cast<MeshShape>(shape));
    emitter_data.setValue("transform", transform.getModel());
    emitter_data.setValue("emission_scaling", mesh_light_component.intensity);
    emitter_data.setValue("texture_emissive", mesh_light_component.getEmissiveTexture());

    atcg::ref_ptr<MeshEmitter> mesh_emitter = atcg::make_ref<MeshEmitter>(emitter_data);
    mesh_emitter->initializePipeline(_pipeline, _sbt);

    auto original_name = entity.getComponent<NameComponent>().name();
    auto new_entity    = result->createEntity(original_name);

    new_entity.addComponent<ShapeComponent>(shape);
    new_entity.addComponent<TransformComponent>(transform);
    new_entity.addComponent<int32_t>((int32_t)entity.entity_handle());
    new_entity.addComponent<EmitterComponent>(mesh_emitter);

    // Add AABB of the shape to the scene AABB
    atcg::BoundingBox shape_aabb = geometry.graph()->getBoundingBox();
    shape_aabb                   = atcg::Utils::transformBoundingBox(shape_aabb, transform.getModel());
    _scene_aabb                  = _scene_aabb + shape_aabb;
}

atcg::ref_ptr<OptixScene>
SceneAdapter::apply(const atcg::ref_ptr<Scene>& scene, const uint32_t width, const uint32_t height)
{
    // Cache shapes and materials
    auto& registry = AssetManager::getAssetRegistry();
    for(auto entry: registry)
    {
        if(entry.second.type == AssetType::Graph)
        {
            auto graph = AssetManager::getAsset<Graph>(entry.first);
            if(graph && graph->type() == GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH)
            {
                atcg::Dictionary shape_dict;
                shape_dict.setValue("mesh", graph);
                atcg::ref_ptr<MeshShape> shape = atcg::make_ref<MeshShape>(shape_dict);
                shape->initializePipeline(_pipeline, _sbt);
                shape->prepareAccelerationStructure(_context);
                _shape_cache.insert(std::make_pair(entry.first, shape));
            }
        }

        if(entry.second.type == AssetType::Material)
        {
            auto material = AssetManager::getAsset<Material>(entry.first);
            if(material)
            {
                material->initializePipeline(_pipeline, _sbt);
                _bsdf_cache.insert(std::make_pair(entry.first, material));
            }
        }
    }

    // Insert default material
    atcg::Dictionary default_material_dict;
    atcg::ref_ptr<Material> material = atcg::MaterialRegistry::createMaterial("Opaque", default_material_dict);
    material->initializePipeline(_pipeline, _sbt);

    _bsdf_cache.insert(std::make_pair(0, material));

    atcg::ref_ptr<OptixScene> result = atcg::make_ref<OptixScene>();
    std::vector<const EmitterVPtrTable*> tables;

    auto light_view = scene->getAllEntitiesWith<atcg::TransformComponent, atcg::PointLightComponent>();
    for(auto e: light_view)
    {
        Entity entity(e, scene.get());

        auto new_entity = result->createEntity();

        auto& point_light_component = entity.getComponent<PointLightComponent>();
        auto& transform             = entity.getComponent<TransformComponent>();

        atcg::Dictionary point_light_data;
        point_light_data.setValue("position", transform.getPosition());
        point_light_data.setValue("color", point_light_component.color);
        point_light_data.setValue("intensity", point_light_component.intensity);
        auto point_light = atcg::make_ref<atcg::PointEmitter>(point_light_data);
        point_light->initializePipeline(_pipeline, _sbt);
        new_entity.addComponent<EmitterComponent>(point_light);
    }


    // Extract scene information
    auto view = scene->getAllEntitiesWith<GeometryComponent, TransformComponent>();
    for(auto e: view)
    {
        Entity entity(e, scene.get());

        prepareComponent<MeshRenderComponent>(result, entity);
        prepareComponent<PointSphereRenderComponent>(result, entity);
        prepareComponent<EdgeCylinderRenderComponent>(result, entity);
        prepareComponent<InstanceRenderComponent>(result, entity);
        prepareComponent<MeshLightComponent>(result, entity);
    }

    auto emitter_view = result->getAllEntitiesWith<EmitterComponent>();
    for(auto e: emitter_view)
    {
        atcg::Entity entity(e, result.get());

        auto emitter = entity.getComponent<EmitterComponent>().emitter;
        result->_emitter.push_back(emitter);
        tables.push_back(emitter->getVPtrTable());
    }

    if(scene->hasSkybox())
    {
        auto skybox_texture = scene->getSkyboxTexture();

        atcg::Dictionary emitter_dict;
        emitter_dict.setValue("environment_texture", skybox_texture);
        emitter_dict.setValue("scene_aabb", _scene_aabb);
        result->_environment_emitter = atcg::make_ref<atcg::EnvironmentEmitter>(emitter_dict);
        result->_environment_emitter->initializePipeline(_pipeline, _sbt);
        tables.push_back(result->_environment_emitter->getVPtrTable());
    }

    result->_emitter_vptr_tables.upload(tables.data(), tables.size());

    auto shape_view = result->getAllEntitiesWith<int32_t>();
    for(auto e: shape_view)
    {
        atcg::Entity entity(e, result.get());

        Dictionary shape_data;
        if(entity.hasComponent<ShapeComponent>())
        {
            shape_data.setValue("shape", entity.getComponent<ShapeComponent>().shape);
        }

        if(entity.hasComponent<BSDFComponent>())
        {
            auto handle = entity.getComponent<BSDFComponent>().bsdf_handle;
            shape_data.setValue("bsdf", _bsdf_cache.at(handle));
        }

        if(entity.hasComponent<TransformComponent>())
        {
            shape_data.setValue("transform", entity.getComponent<TransformComponent>().getModel());
        }

        if(entity.hasComponent<MediumComponent>())
        {
            shape_data.setValue("inside_medium", entity.getComponent<MediumComponent>().medium());
        }

        if(entity.hasComponent<EmitterComponent>())
        {
            shape_data.setValue("emitter", entity.getComponent<EmitterComponent>().emitter);
        }

        if(entity.hasComponent<glm::vec3>())
        {
            shape_data.setValue("color", entity.getComponent<glm::vec3>());
        }

        if(entity.hasComponent<int32_t>())
        {
            shape_data.setValue("entity_id", entity.getComponent<int32_t>());
        }

        auto shape = atcg::make_ref<ShapeInstance>(shape_data);
        shape->initializePipeline(_pipeline, _sbt);

        result->_shapes.push_back(shape);
    }


    result->_ias = atcg::make_ref<InstanceAccelerationStructure>(_context, result->_shapes, _pipeline->numRays());

    if(scene->getCamera())
    {
        atcg::Dictionary film_dict;
        film_dict.setValue("width", width);
        film_dict.setValue("height", height);
        atcg::ref_ptr<Film> film = atcg::make_ref<HDRFilm>(film_dict);

        atcg::Dictionary sensor_dict;
        sensor_dict.setValue("film", film);
        sensor_dict.setValue<atcg::ref_ptr<Camera>>("camera", scene->getCamera());
        result->_sensor = atcg::make_ref<PinholeCamera>(sensor_dict);

        result->_sensor->initializePipeline(_pipeline, _sbt);
    }
    else
    {
        ATCG_WARN("No camera found in scene!");
        result->_sensor = nullptr;
    }

    return result;
}
}    // namespace atcg