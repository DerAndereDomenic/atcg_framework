#include <Integrator/PathtracingIntegrator.h>

#include <Core/Path.h>
#include <Core/Assert.h>
#include <Core/CUDA.h>
#include <Core/Common.h>
#include <Scene/Components.h>
#include <Scene/Entity.h>
#include <Shape/Shape.h>
#include <Shape/ShapeInstance.h>
#include <Shape/MeshShape.h>
#include <BSDF/PBRBSDF.h>
#include <DataStructure/WorkerPool.h>
#include <Emitter/MeshEmitter.h>

#include <optix_stubs.h>

namespace atcg
{
PathtracingIntegrator::PathtracingIntegrator(const atcg::ref_ptr<RaytracingContext>& context, const Dictionary& dict)
    : Integrator(context, dict)
{
}

PathtracingIntegrator::~PathtracingIntegrator() {}

template<typename T>
void PathtracingIntegrator::prepareComponent(Entity entity,
                                             const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                             const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
}

template<>
void PathtracingIntegrator::prepareComponent<MeshRenderComponent>(Entity entity,
                                                                  const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                                                  const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    if(!entity.hasComponent<MeshRenderComponent>()) return;

    MeshRenderComponent& component = entity.getComponent<MeshRenderComponent>();

    if(!component.visible) return;

    auto& transform = entity.getComponent<TransformComponent>();
    auto& material  = component.material;

    auto graph = entity.getComponent<GeometryComponent>().graph;
    atcg::Dictionary shape_dict;
    shape_dict.setValue("mesh", graph);
    atcg::ref_ptr<Shape> shape = atcg::make_ref<MeshShape>(shape_dict);
    shape->initializePipeline(pipeline, sbt);
    shape->prepareAccelerationStructure(_context);

    atcg::Dictionary bsdf_dict;
    bsdf_dict.setValue("material", material);
    atcg::ref_ptr<BSDF> bsdf = atcg::make_ref<PBRBSDF>(bsdf_dict);
    bsdf->initializePipeline(pipeline, sbt);

    atcg::ref_ptr<Emitter> mesh_emitter = nullptr;
    if(material.emissive)
    {
        Dictionary emitter_data;
        emitter_data.setValue<atcg::ref_ptr<MeshShape>>("shape", std::dynamic_pointer_cast<MeshShape>(shape));
        emitter_data.setValue("transform", transform.getModel());
        emitter_data.setValue("emission_scaling", material.emission_scale);
        emitter_data.setValue("texture_emissive", material.getEmissiveTexture());

        mesh_emitter = atcg::make_ref<MeshEmitter>(emitter_data);
        mesh_emitter->initializePipeline(pipeline, sbt);

        _emitter.push_back(mesh_emitter);
    }

    Dictionary shape_data;
    shape_data.setValue("shape", shape);
    shape_data.setValue("bsdf", bsdf);
    shape_data.setValue("transform", transform.getModel());
    shape_data.setValue<int32_t>("entity_id", (int32_t)entity.entity_handle());
    shape_data.setValue("emitter", mesh_emitter);
    auto shape_instance = atcg::make_ref<ShapeInstance>(shape_data);
    shape_instance->initializePipeline(pipeline, sbt);

    _shapes.push_back(shape_instance);
}

template<>
void PathtracingIntegrator::prepareComponent<PointSphereRenderComponent>(
    Entity entity,
    const atcg::ref_ptr<RayTracingPipeline>& pipeline,
    const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    if(!entity.hasComponent<PointSphereRenderComponent>()) return;

    PointSphereRenderComponent& component = entity.getComponent<PointSphereRenderComponent>();

    if(!component.visible) return;

    auto& transform            = entity.getComponent<TransformComponent>();
    glm::mat4 global_transform = transform.getModel();
    auto& material             = component.material;

    auto graph = atcg::IO::read_mesh((atcg::resource_directory() / "sphere_low.obj").string());
    atcg::Dictionary shape_dict;
    shape_dict.setValue("mesh", graph);
    atcg::ref_ptr<Shape> shape = atcg::make_ref<MeshShape>(shape_dict);
    shape->initializePipeline(pipeline, sbt);
    shape->prepareAccelerationStructure(_context);

    atcg::Dictionary bsdf_dict;
    bsdf_dict.setValue("material", material);
    atcg::ref_ptr<BSDF> bsdf = atcg::make_ref<PBRBSDF>(bsdf_dict);
    bsdf->initializePipeline(pipeline, sbt);

    auto mesh            = entity.getComponent<GeometryComponent>().graph;
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
        shape_instance->initializePipeline(pipeline, sbt);
    }

    _shapes.insert(_shapes.end(), new_shapes.begin(), new_shapes.end());

    mesh->unmapAllHostPointers();
}

template<>
void PathtracingIntegrator::prepareComponent<EdgeCylinderRenderComponent>(
    Entity entity,
    const atcg::ref_ptr<RayTracingPipeline>& pipeline,
    const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    if(!entity.hasComponent<EdgeCylinderRenderComponent>()) return;

    EdgeCylinderRenderComponent& component = entity.getComponent<EdgeCylinderRenderComponent>();

    if(!component.visible) return;

    auto& transform            = entity.getComponent<TransformComponent>();
    glm::mat4 global_transform = transform.getModel();
    auto& material             = component.material;

    auto graph = atcg::IO::read_mesh((atcg::resource_directory() / "cylinder.obj").string());
    atcg::Dictionary shape_dict;
    shape_dict.setValue("mesh", graph);
    atcg::ref_ptr<Shape> shape = atcg::make_ref<MeshShape>(shape_dict);
    shape->initializePipeline(pipeline, sbt);
    shape->prepareAccelerationStructure(_context);

    atcg::Dictionary bsdf_dict;
    bsdf_dict.setValue("material", material);
    atcg::ref_ptr<BSDF> bsdf = atcg::make_ref<PBRBSDF>(bsdf_dict);
    bsdf->initializePipeline(pipeline, sbt);

    auto mesh            = entity.getComponent<GeometryComponent>().graph;
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
        shape_instance->initializePipeline(pipeline, sbt);
    }

    _shapes.insert(_shapes.end(), new_shapes.begin(), new_shapes.end());

    mesh->unmapAllHostPointers();
}

template<>
void PathtracingIntegrator::prepareComponent<InstanceRenderComponent>(Entity entity,
                                                                      const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                                                      const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    if(!entity.hasComponent<InstanceRenderComponent>()) return;

    InstanceRenderComponent& component = entity.getComponent<InstanceRenderComponent>();

    if(!component.visible) return;

    if(component.instance_vbos.size() != 2)
    {
        ATCG_WARN("Expecting size two for number of instances in path tracer...");
        return;
    }

    auto& transform            = entity.getComponent<TransformComponent>();
    glm::mat4 global_transform = transform.getModel();
    auto& material             = component.material;

    auto graph = entity.getComponent<GeometryComponent>().graph;
    atcg::Dictionary shape_dict;
    shape_dict.setValue("mesh", graph);
    atcg::ref_ptr<Shape> shape = atcg::make_ref<MeshShape>(shape_dict);
    shape->initializePipeline(pipeline, sbt);
    shape->prepareAccelerationStructure(_context);

    atcg::Dictionary bsdf_dict;
    bsdf_dict.setValue("material", material);
    atcg::ref_ptr<BSDF> bsdf = atcg::make_ref<PBRBSDF>(bsdf_dict);
    bsdf->initializePipeline(pipeline, sbt);

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
        shape_instance->initializePipeline(pipeline, sbt);
    }

    _shapes.insert(_shapes.end(), new_shapes.begin(), new_shapes.end());

    transform_vbo->unmapHostPointers();
    color_vbo->unmapHostPointers();
}

void PathtracingIntegrator::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                               const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    std::vector<const EmitterVPtrTable*> tables;
    if(_scene->hasSkybox())
    {
        auto skybox_texture = _scene->getSkyboxTexture();

        atcg::Dictionary emitter_dict;
        emitter_dict.setValue("environment_texture", skybox_texture);
        _environment_emitter = atcg::make_ref<atcg::EnvironmentEmitter>(emitter_dict);
        _environment_emitter->initializePipeline(pipeline, sbt);
        tables.push_back(_environment_emitter->getVPtrTable());
    }

    auto light_view = _scene->getAllEntitiesWith<atcg::TransformComponent, atcg::PointLightComponent>();
    for(auto e: light_view)
    {
        Entity entity(e, _scene.get());

        auto& point_light_component = entity.getComponent<PointLightComponent>();
        auto& transform             = entity.getComponent<TransformComponent>();

        atcg::Dictionary point_light_data;
        point_light_data.setValue("position", transform.getPosition());
        point_light_data.setValue("color", point_light_component.color);
        point_light_data.setValue("intensity", point_light_component.intensity);
        auto point_light = atcg::make_ref<atcg::PointEmitter>(point_light_data);
        point_light->initializePipeline(pipeline, sbt);
        _emitter.push_back(point_light);
    }


    // Extract scene information
    auto view = _scene->getAllEntitiesWith<GeometryComponent, TransformComponent>();
    for(auto e: view)
    {
        Entity entity(e, _scene.get());

        prepareComponent<MeshRenderComponent>(entity, pipeline, sbt);
        prepareComponent<PointSphereRenderComponent>(entity, pipeline, sbt);
        prepareComponent<EdgeCylinderRenderComponent>(entity, pipeline, sbt);
        prepareComponent<InstanceRenderComponent>(entity, pipeline, sbt);
    }

    // No all the emitters are initialized
    for(auto emitter: _emitter)
    {
        tables.push_back(emitter->getVPtrTable());
    }
    _emitters.upload(tables.data(), tables.size());

    const std::string ptx_raygen_filename = "./bin/PathtracingIntegrator_ptx.ptx";
    OptixProgramGroup raygen_prog_group   = pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__rg"});
    OptixProgramGroup miss_prog_group     = pipeline->addMissShader({ptx_raygen_filename, "__miss__ms"});
    OptixProgramGroup occl_prog_group     = pipeline->addMissShader({ptx_raygen_filename, "__miss__occlusion"});

    _raygen_index         = sbt->addRaygenEntry(raygen_prog_group);
    _surface_miss_index   = sbt->addMissEntry(miss_prog_group);
    _occlusion_miss_index = sbt->addMissEntry(occl_prog_group);

    _ias = atcg::make_ref<InstanceAccelerationStructure>(_context, _shapes);

    _pipeline = pipeline;
    _sbt      = sbt;
}

void PathtracingIntegrator::reset()
{
    _frame_counter = 0;
}

void PathtracingIntegrator::generateRays(Dictionary& in_out_dictionary)
{
    auto camera     = in_out_dictionary.getValue<atcg::ref_ptr<atcg::PerspectiveCamera>>("camera");
    auto output     = in_out_dictionary.getValue<torch::Tensor>("output");
    auto entity_ids = in_out_dictionary.getValueOr<torch::Tensor>("entity_ids", torch::empty({0}));

    if(_accumulation_buffer.numel() == 0 || _frame_counter == 0 || _accumulation_buffer.size(0) != output.size(0) ||
       _accumulation_buffer.size(1) != output.size(1))
    {
        _accumulation_buffer =
            torch::zeros({output.size(0), output.size(1), 3}, atcg::TensorOptions::floatDeviceOptions());
    }

    PathtracingParams params;

    glm::mat4 inv_camera_view = glm::inverse(camera->getView());
    memcpy(params.cam_eye, glm::value_ptr(inv_camera_view[3]), sizeof(glm::vec3));
    memcpy(params.U, glm::value_ptr(glm::normalize(inv_camera_view[0])), sizeof(glm::vec3));
    memcpy(params.V, glm::value_ptr(glm::normalize(inv_camera_view[1])), sizeof(glm::vec3));
    memcpy(params.W, glm::value_ptr(-glm::normalize(inv_camera_view[2])), sizeof(glm::vec3));
    params.fov_y = camera->getFOV();

    params.output_image = (glm::u8vec4*)output.data_ptr();
    params.image_height = output.size(0);
    params.image_width  = output.size(1);
    params.handle       = _ias->getTraversableHandle();

    params.entity_ids = entity_ids.numel() > 0 ? (int32_t*)entity_ids.data_ptr() : nullptr;

    params.accumulation_buffer = (glm::vec3*)_accumulation_buffer.data_ptr();

    params.frame_counter = _frame_counter++;

    params.num_emitters        = _emitters.size();
    params.emitters            = _emitters.get();
    params.environment_emitter = _environment_emitter ? _environment_emitter->getVPtrTable() : nullptr;

    params.surface_trace_params.rayFlags     = OPTIX_RAY_FLAG_NONE;
    params.surface_trace_params.SBToffset    = 0;
    params.surface_trace_params.SBTstride    = 1;
    params.surface_trace_params.missSBTIndex = _surface_miss_index;

    params.occlusion_trace_params.rayFlags  = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
    params.occlusion_trace_params.SBToffset = 0;
    params.occlusion_trace_params.SBTstride = 1;
    params.occlusion_trace_params.missSBTIndex = _occlusion_miss_index;

    _launch_params.upload(&params);

    OPTIX_CHECK(optixLaunch(_pipeline->getPipeline(),
                            nullptr,
                            (CUdeviceptr)_launch_params.get(),
                            sizeof(PathtracingParams),
                            _sbt->getSBT(_raygen_index),
                            output.size(1),
                            output.size(0),
                            1));    // depth

    CUDA_SAFE_CALL(cudaStreamSynchronize(nullptr));
}
}    // namespace atcg