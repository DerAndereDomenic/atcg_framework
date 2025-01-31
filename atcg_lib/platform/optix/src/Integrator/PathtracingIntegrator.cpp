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
PathtracingIntegrator::PathtracingIntegrator(OptixDeviceContext context) : Integrator(context) {}

PathtracingIntegrator::~PathtracingIntegrator() {}

void PathtracingIntegrator::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                               const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
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

        auto bsdf = atcg::make_ref<PBRBSDF>(material);
        bsdf->initializePipeline(pipeline, sbt);

        auto shape_instance = atcg::make_ref<ShapeInstance>(shape, bsdf, transform.getModel());
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

    _ias = atcg::make_ref<IAS>(_context, _shapes);

    _pipeline = pipeline;
    _sbt      = sbt;
}

void PathtracingIntegrator::generateRays(const atcg::ref_ptr<PerspectiveCamera>& camera, torch::Tensor& output)
{
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