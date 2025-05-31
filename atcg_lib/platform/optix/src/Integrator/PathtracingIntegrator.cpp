#include <Integrator/PathtracingIntegrator.h>

#include <Core/CUDA.h>
#include <Core/Common.h>
#include <Scene/Components.h>
#include <Scene/Entity.h>
#include <Shape/Shape.h>
#include <Shape/ShapeInstance.h>
#include <Shape/MeshShape.h>
#include <BSDF/PBRBSDF.h>

#include <optix_stubs.h>

namespace atcg
{
PathtracingIntegrator::PathtracingIntegrator(const atcg::ref_ptr<RaytracingContext>& context) : Integrator(context) {}

PathtracingIntegrator::~PathtracingIntegrator() {}

void PathtracingIntegrator::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                               const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    std::vector<const EmitterVPtrTable*> tables;
    if(_scene->hasSkybox())
    {
        auto skybox_texture = _scene->getSkyboxTexture();

        _environment_emitter = atcg::make_ref<atcg::EnvironmentEmitter>(skybox_texture);
        _environment_emitter->initializePipeline(pipeline, sbt);
        tables.push_back(_environment_emitter->getVPtrTable());
    }

    auto light_view = _scene->getAllEntitiesWith<atcg::TransformComponent, atcg::PointLightComponent>();
    for(auto e: light_view)
    {
        Entity entity(e, _scene.get());

        auto& point_light_component = entity.getComponent<PointLightComponent>();
        auto& transform             = entity.getComponent<TransformComponent>();

        auto point_light = atcg::make_ref<atcg::PointEmitter>(transform.getPosition(), point_light_component);
        point_light->initializePipeline(pipeline, sbt);
        _emitter.push_back(point_light);
        tables.push_back(point_light->getVPtrTable());
    }

    _emitters.upload(tables.data(), tables.size());

    // Extract scene information
    auto view = _scene->getAllEntitiesWith<GeometryComponent, MeshRenderComponent, TransformComponent>();
    for(auto e: view)
    {
        Entity entity(e, _scene.get());

        auto& transform = entity.getComponent<TransformComponent>();
        auto& material  = entity.getComponent<MeshRenderComponent>().material;

        auto graph                 = entity.getComponent<GeometryComponent>().graph;
        atcg::ref_ptr<Shape> shape = atcg::make_ref<MeshShape>(graph);
        shape->initializePipeline(pipeline, sbt);
        shape->prepareAccelerationStructure(_context);

        atcg::ref_ptr<BSDF> bsdf = atcg::make_ref<PBRBSDF>(material);
        bsdf->initializePipeline(pipeline, sbt);

        Dictionary shape_data;
        shape_data.setValue("shape", shape);
        shape_data.setValue("bsdf", bsdf);
        shape_data.setValue("transform", transform.getModel());
        auto shape_instance = atcg::make_ref<ShapeInstance>(shape_data);
        shape_instance->initializePipeline(pipeline, sbt);

        _shapes.push_back(shape_instance);
    }

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
    auto camera = in_out_dictionary.getValue<atcg::ref_ptr<atcg::PerspectiveCamera>>("camera");
    auto output = in_out_dictionary.getValue<torch::Tensor>("output");

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