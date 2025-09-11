#include <Scene/SceneAdapter.h>
#include <Scene/Components.h>
#include <Emitter/PointEmitter.h>
#include <Emitter/MeshEmitter.h>
#include <DataStructure/WorkerPool.h>
#include <Shape/MeshShape.h>
#include <Core/Path.h>
#include <Core/Assert.h>
#include <BSDF/BSDFFactory.h>

namespace atcg
{
template<typename T>
void SceneAdapter::prepareComponent(const atcg::ref_ptr<OptixScene>& result, Entity entity)
{
}

template<>
void SceneAdapter::prepareComponent<MeshRenderComponent>(const atcg::ref_ptr<OptixScene>& result, Entity entity)
{
    if(!entity.hasComponent<MeshRenderComponent>()) return;
    if(!entity.hasComponent<GeometryComponent>()) return;

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

    atcg::ref_ptr<Emitter> mesh_emitter = nullptr;
    if(entity.hasComponent<MeshLightComponent>())
    {
        auto& mesh_light_component = entity.getComponent<MeshLightComponent>();
        Dictionary emitter_data;
        emitter_data.setValue<atcg::ref_ptr<MeshShape>>("shape", std::dynamic_pointer_cast<MeshShape>(shape));
        emitter_data.setValue("transform", transform.getModel());
        emitter_data.setValue("emission_scaling", mesh_light_component.intensity);
        emitter_data.setValue("texture_emissive", mesh_light_component.getEmissiveTexture());

        mesh_emitter = atcg::make_ref<MeshEmitter>(emitter_data);
        mesh_emitter->initializePipeline(_pipeline, _sbt);

        result->_emitter.push_back(mesh_emitter);
    }

    Dictionary shape_data;
    shape_data.setValue("shape", shape);
    shape_data.setValue("bsdf", bsdf);
    shape_data.setValue("transform", transform.getModel());
    shape_data.setValue<int32_t>("entity_id", (int32_t)entity.entity_handle());
    shape_data.setValue("emitter", mesh_emitter);
    auto shape_instance = atcg::make_ref<ShapeInstance>(shape_data);
    shape_instance->initializePipeline(_pipeline, _sbt);

    result->_shapes.push_back(shape_instance);
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
    auto& material             = component.material();

    auto graph = atcg::IO::read_mesh((atcg::resource_directory() / "sphere_low.obj").string());
    atcg::Dictionary shape_dict;
    shape_dict.setValue("mesh", graph);
    atcg::ref_ptr<Shape> shape = atcg::make_ref<MeshShape>(shape_dict);
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

    std::vector<atcg::ref_ptr<ShapeInstance>> new_shapes(n_instances);

    atcg::WorkerPool pool(32);
    pool.start();

    for(int i = 0; i < n_instances; ++i)
    {
        pool.pushJob(
            [offsets,
             colors,
             i,
             &global_transform,
             &inv_scale_model,
             &scale_primitive,
             shape,
             bsdf,
             entity,
             &new_shapes]()
            {
                glm::vec3 offset =
                    glm::vec3(offsets[i][0].item<float>(), offsets[i][1].item<float>(), offsets[i][2].item<float>());
                glm::vec3 color =
                    glm::vec3(colors[i][0].item<float>(), colors[i][1].item<float>(), colors[i][2].item<float>());
                glm::mat4 total_transform = glm::translate(glm::vec3(global_transform * glm::vec4(offset, 0))) *
                                            global_transform * inv_scale_model * scale_primitive;

                Dictionary shape_data;
                shape_data.setValue("shape", shape);
                shape_data.setValue("bsdf", bsdf);
                shape_data.setValue("transform", total_transform);
                shape_data.setValue<int32_t>("entity_id", (int32_t)entity.entity_handle());
                shape_data.setValue("color", color);
                auto shape_instance = atcg::make_ref<ShapeInstance>(shape_data);

                new_shapes[i] = shape_instance;
            });
    }

    pool.waitDone();

    // Not thread safe
    for(auto shape_instance: new_shapes)
    {
        shape_instance->initializePipeline(_pipeline, _sbt);
    }

    result->_shapes.insert(result->_shapes.end(), new_shapes.begin(), new_shapes.end());

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
    auto& material             = component.material();

    auto graph = atcg::IO::read_mesh((atcg::resource_directory() / "cylinder.obj").string());
    atcg::Dictionary shape_dict;
    shape_dict.setValue("mesh", graph);
    atcg::ref_ptr<Shape> shape = atcg::make_ref<MeshShape>(shape_dict);
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

    std::vector<atcg::ref_ptr<ShapeInstance>> new_shapes(n_instances);

    atcg::WorkerPool pool(32);
    pool.start();

    // Logic from cylinder_edge.vs
    for(int i = 0; i < n_instances; ++i)
    {
        pool.pushJob(
            [indices, positions, i, &global_transform, &component, shape, bsdf, entity, &new_shapes]()
            {
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

                Dictionary shape_data;
                shape_data.setValue("shape", shape);
                shape_data.setValue("bsdf", bsdf);
                shape_data.setValue("transform", model_edge);
                shape_data.setValue<int32_t>("entity_id", (int32_t)entity.entity_handle());
                shape_data.setValue("color", edge_color);
                auto shape_instance = atcg::make_ref<ShapeInstance>(shape_data);

                new_shapes[i] = shape_instance;
            });
    }

    pool.waitDone();

    // Not thread safe
    for(auto shape_instance: new_shapes)
    {
        shape_instance->initializePipeline(_pipeline, _sbt);
    }

    result->_shapes.insert(result->_shapes.end(), new_shapes.begin(), new_shapes.end());

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
    auto& material             = component.material();

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

    std::vector<atcg::ref_ptr<ShapeInstance>> new_shapes(n_instances);

    atcg::WorkerPool pool(32);
    pool.start();

    for(int i = 0; i < n_instances; ++i)
    {
        pool.pushJob(
            [shape, bsdf, &global_transform, i, transforms, colors, &new_shapes, entity]()
            {
                Dictionary shape_data;
                shape_data.setValue("shape", shape);
                shape_data.setValue("bsdf", bsdf);
                shape_data.setValue("transform", global_transform * transforms[i]);
                shape_data.setValue<int32_t>("entity_id", (int32_t)entity.entity_handle());
                shape_data.setValue<glm::vec3>("color", glm::vec3(colors[i]));
                auto shape_instance = atcg::make_ref<ShapeInstance>(shape_data);

                new_shapes[i] = shape_instance;
            });
    }

    pool.waitDone();

    // Not thread safe
    for(auto shape_instance: new_shapes)
    {
        shape_instance->initializePipeline(_pipeline, _sbt);
    }

    result->_shapes.insert(result->_shapes.end(), new_shapes.begin(), new_shapes.end());

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

    atcg::ref_ptr<Emitter> mesh_emitter = nullptr;
    auto& mesh_light_component          = entity.getComponent<MeshLightComponent>();
    Dictionary emitter_data;
    emitter_data.setValue<atcg::ref_ptr<MeshShape>>("shape", std::dynamic_pointer_cast<MeshShape>(shape));
    emitter_data.setValue("transform", transform.getModel());
    emitter_data.setValue("emission_scaling", mesh_light_component.intensity);
    emitter_data.setValue("texture_emissive", mesh_light_component.getEmissiveTexture());

    mesh_emitter = atcg::make_ref<MeshEmitter>(emitter_data);
    mesh_emitter->initializePipeline(_pipeline, _sbt);

    result->_emitter.push_back(mesh_emitter);


    Dictionary shape_data;
    shape_data.setValue("shape", shape);
    shape_data.setValue("transform", transform.getModel());
    shape_data.setValue<int32_t>("entity_id", (int32_t)entity.entity_handle());
    shape_data.setValue("emitter", mesh_emitter);
    auto shape_instance = atcg::make_ref<ShapeInstance>(shape_data);
    shape_instance->initializePipeline(_pipeline, _sbt);

    result->_shapes.push_back(shape_instance);
}

atcg::ref_ptr<OptixScene> SceneAdapter::apply(const atcg::ref_ptr<Scene>& scene)
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
                atcg::ref_ptr<Shape> shape = atcg::make_ref<MeshShape>(shape_dict);
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
                atcg::Dictionary bsdf_dict;
                bsdf_dict.setValue("material", material);
                atcg::ref_ptr<BSDF> bsdf = BSDFFactory::createBSDF(material->getMaterialType(), bsdf_dict);
                bsdf->initializePipeline(_pipeline, _sbt);

                _bsdf_cache.insert(std::make_pair(entry.first, bsdf));
            }
        }
    }

    // Insert default material
    atcg::ref_ptr<Material> material = atcg::make_ref<Material>();
    atcg::Dictionary bsdf_dict;
    bsdf_dict.setValue("material", material);
    atcg::ref_ptr<BSDF> bsdf = BSDFFactory::createBSDF(material->getMaterialType(), bsdf_dict);
    bsdf->initializePipeline(_pipeline, _sbt);

    _bsdf_cache.insert(std::make_pair(0, bsdf));

    atcg::ref_ptr<OptixScene> result = atcg::make_ref<OptixScene>();
    std::vector<const EmitterVPtrTable*> tables;
    if(scene->hasSkybox())
    {
        auto skybox_texture = scene->getSkyboxTexture();

        atcg::Dictionary emitter_dict;
        emitter_dict.setValue("environment_texture", skybox_texture);
        result->_environment_emitter = atcg::make_ref<atcg::EnvironmentEmitter>(emitter_dict);
        result->_environment_emitter->initializePipeline(_pipeline, _sbt);
        tables.push_back(result->_environment_emitter->getVPtrTable());
    }

    auto light_view = scene->getAllEntitiesWith<atcg::TransformComponent, atcg::PointLightComponent>();
    for(auto e: light_view)
    {
        Entity entity(e, scene.get());

        auto& point_light_component = entity.getComponent<PointLightComponent>();
        auto& transform             = entity.getComponent<TransformComponent>();

        atcg::Dictionary point_light_data;
        point_light_data.setValue("position", transform.getPosition());
        point_light_data.setValue("color", point_light_component.color);
        point_light_data.setValue("intensity", point_light_component.intensity);
        auto point_light = atcg::make_ref<atcg::PointEmitter>(point_light_data);
        point_light->initializePipeline(_pipeline, _sbt);
        result->_emitter.push_back(point_light);
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

    // Now all the emitters are initialized
    for(auto emitter: result->_emitter)
    {
        tables.push_back(emitter->getVPtrTable());
    }
    result->_emitter_vptr_tables.upload(tables.data(), tables.size());

    result->_ias = atcg::make_ref<InstanceAccelerationStructure>(_context, result->_shapes);

    return result;
}
}    // namespace atcg