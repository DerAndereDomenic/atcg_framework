#include "DiffPathtracingIntegrator.h"

#include <Core/Path.h>
#include <Core/Assert.h>
#include <Core/CUDA.h>
#include <Core/Common.h>
#include <Scene/Components.h>
#include <Scene/Entity.h>
#include <Shape/Shape.h>
#include <Shape/ShapeInstance.h>
#include <Shape/MeshShape.h>
#include <DataStructure/WorkerPool.h>
#include <Emitter/MeshEmitter.h>
#include <Scene/SceneAdapter.h>

#include <optix_stubs.h>

namespace atcg
{
DiffPathtracingIntegrator::DiffPathtracingIntegrator(const atcg::ref_ptr<RaytracingContext>& context,
                                                     const Dictionary& dict)
    : Integrator(context, dict)
{
}

DiffPathtracingIntegrator::~DiffPathtracingIntegrator() {}

void DiffPathtracingIntegrator::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                                   const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_raygen_filename       = "./bin/DiffPathtracingIntegrator_ptx.ptx";
    OptixProgramGroup raygen_prog_group_forward = pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__forward"});
    OptixProgramGroup raygen_prog_group_backward =
        pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__backward"});
    OptixProgramGroup miss_prog_group = pipeline->addMissShader({ptx_raygen_filename, "__miss__ms"});
    OptixProgramGroup occl_prog_group = pipeline->addMissShader({ptx_raygen_filename, "__miss__occlusion"});

    _raygen_index_forward  = sbt->addRaygenEntry(raygen_prog_group_forward);
    _raygen_index_backward = sbt->addRaygenEntry(raygen_prog_group_backward);
    _surface_miss_index    = sbt->addMissEntry(miss_prog_group);
    _occlusion_miss_index  = sbt->addMissEntry(occl_prog_group);

    _pipeline = pipeline;
    _sbt      = sbt;

    _optix_scene = SceneAdapter(_context, pipeline, sbt).apply(_scene);

    _differentiable_components.clear();
    for(auto shape: _optix_scene->getShapes())
    {
        auto diff = std::dynamic_pointer_cast<Differentiable>(shape->getBSDF());
        if(diff)
        {
            _differentiable_components.push_back(diff.get());
        }
    }

    ATCG_TRACE("Number differentiable objects: {}", _differentiable_components.size());
}

void DiffPathtracingIntegrator::onImGuiRender()
{
    _panel.renderPanel(_optix_scene);
}

void DiffPathtracingIntegrator::reset()
{
    _frame_counter = 0;
}

torch::Tensor DiffPathtracingIntegrator::getHDR() const
{
    return _accumulation_buffer.clone();
}

void DiffPathtracingIntegrator::_forwardTrace(Dictionary& in_out_dictionary)
{
    auto camera     = in_out_dictionary.getValue<atcg::ref_ptr<atcg::PerspectiveCamera>>("camera");
    uint32_t width  = in_out_dictionary.getValue<uint32_t>("width");
    uint32_t height = in_out_dictionary.getValue<uint32_t>("height");
    auto entity_ids = in_out_dictionary.getValueOr<torch::Tensor>("entity_ids", torch::empty({0}));

    if(_accumulation_buffer.numel() == 0 || _frame_counter == 0 || _accumulation_buffer.size(0) != height ||
       _accumulation_buffer.size(1) != width)
    {
        _accumulation_buffer = torch::zeros({height, width, 3}, atcg::TensorOptions::floatDeviceOptions());
        _current_sample      = torch::zeros({height, width, 3}, atcg::TensorOptions::floatDeviceOptions());
    }

    DiffPathtracingParams params;

    glm::mat4 inv_camera_view = glm::inverse(camera->getView());
    memcpy(params.cam_eye, glm::value_ptr(inv_camera_view[3]), sizeof(glm::vec3));
    memcpy(params.U, glm::value_ptr(glm::normalize(inv_camera_view[0])), sizeof(glm::vec3));
    memcpy(params.V, glm::value_ptr(glm::normalize(inv_camera_view[1])), sizeof(glm::vec3));
    memcpy(params.W, glm::value_ptr(-glm::normalize(inv_camera_view[2])), sizeof(glm::vec3));
    params.fov_y = camera->getFOV();

    params.image_height = height;
    params.image_width  = width;
    params.handle       = _optix_scene->getIAS()->getTraversableHandle();

    params.entity_ids = entity_ids.numel() > 0 ? (int32_t*)entity_ids.data_ptr() : nullptr;

    params.accumulation_buffer = (glm::vec3*)_accumulation_buffer.data_ptr();
    params.current_sample      = (glm::vec3*)_current_sample.data_ptr();

    params.rng_index     = _iteration_counter + _frame_counter;
    params.frame_counter = _frame_counter++;

    params.num_emitters        = _optix_scene->getEmitterVPtrTables().size();
    params.emitters            = _optix_scene->getEmitterVPtrTables().get();
    auto environment_emitter   = _optix_scene->getEnvironmentEmitter();
    params.environment_emitter = environment_emitter ? environment_emitter->getVPtrTable() : nullptr;

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
                            sizeof(DiffPathtracingParams),
                            _sbt->getSBT(_raygen_index_forward),
                            width,
                            height,
                            1));    // depth

    CUDA_SAFE_CALL(cudaStreamSynchronize(nullptr));
}

void DiffPathtracingIntegrator::_backwardTrace(Dictionary& in_out_dictionary)
{
    auto camera     = in_out_dictionary.getValue<atcg::ref_ptr<atcg::PerspectiveCamera>>("camera");
    uint32_t width  = in_out_dictionary.getValue<uint32_t>("width");
    uint32_t height = in_out_dictionary.getValue<uint32_t>("height");
    auto entity_ids = in_out_dictionary.getValueOr<torch::Tensor>("entity_ids", torch::empty({0}));
    auto adjoint_y  = in_out_dictionary.getValue<torch::Tensor>("adjoint_y");
    uint32_t step   = in_out_dictionary.getValue<uint32_t>("step");

    DiffPathtracingParams params;

    glm::mat4 inv_camera_view = glm::inverse(camera->getView());
    memcpy(params.cam_eye, glm::value_ptr(inv_camera_view[3]), sizeof(glm::vec3));
    memcpy(params.U, glm::value_ptr(glm::normalize(inv_camera_view[0])), sizeof(glm::vec3));
    memcpy(params.V, glm::value_ptr(glm::normalize(inv_camera_view[1])), sizeof(glm::vec3));
    memcpy(params.W, glm::value_ptr(-glm::normalize(inv_camera_view[2])), sizeof(glm::vec3));
    params.fov_y = camera->getFOV();

    params.image_height = height;
    params.image_width  = width;
    params.handle       = _optix_scene->getIAS()->getTraversableHandle();

    params.entity_ids = entity_ids.numel() > 0 ? (int32_t*)entity_ids.data_ptr() : nullptr;

    params.accumulation_buffer = (glm::vec3*)_samples[step].data_ptr();    // Input L
    params.adjoint_y           = (glm::vec3*)adjoint_y.data_ptr();         // Input 𝛿L

    params.rng_index     = _iteration_counter + _frame_counter;
    params.frame_counter = _frame_counter++;

    params.num_emitters        = _optix_scene->getEmitterVPtrTables().size();
    params.emitters            = _optix_scene->getEmitterVPtrTables().get();
    auto environment_emitter   = _optix_scene->getEnvironmentEmitter();
    params.environment_emitter = environment_emitter ? environment_emitter->getVPtrTable() : nullptr;

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
                            sizeof(DiffPathtracingParams),
                            _sbt->getSBT(_raygen_index_backward),
                            width,
                            height,
                            1));    // depth

    CUDA_SAFE_CALL(cudaStreamSynchronize(nullptr));
}

void DiffPathtracingIntegrator::generateRays(Dictionary& in_out_dictionary)
{
    auto output_img = in_out_dictionary.getValue<torch::Tensor>("output_img");
    in_out_dictionary.setValue("height", (uint32_t)output_img.size(0));
    in_out_dictionary.setValue("width", (uint32_t)output_img.size(1));
    _forwardTrace(in_out_dictionary);

    // Perform tonemapping here for output display:
    torch::Tensor tonemapped = torch::pow(1.0f - torch::exp(-_accumulation_buffer), 1.0 / 2.4f);

    tonemapped.clamp_(0.0f, 1.0f);
    output_img.fill_(255);
    output_img.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
                          (tonemapped * 255.0f).to(torch::kUInt8));
}

void DiffPathtracingIntegrator::forwardPass(Dictionary& in_out_dictionary)
{
    const uint32_t num_samples = in_out_dictionary.getValueOr<uint32_t>("num_samples", 16);

    reset();
    _samples.clear();
    for(int i = 0; i < num_samples; ++i)
    {
        _forwardTrace(in_out_dictionary);
        _samples.push_back(_current_sample.clone());
    }
    _state = std::move(in_out_dictionary);    // Store for backward pass (cant be stored in ctx directly)
}

void DiffPathtracingIntegrator::backwardPass(const torch::Tensor& adjoint_y)
{
    const uint32_t num_samples = _state.getValueOr<uint32_t>("num_samples", 16);
    _state.setValue("adjoint_y", adjoint_y);

    reset();
    for(uint32_t i = 0; i < num_samples; ++i)
    {
        _state.setValue("step", i);
        _backwardTrace(_state);
    }

    _iteration_counter += num_samples;
}

std::vector<torch::Tensor> DiffPathtracingIntegrator::getParameters() const
{
    std::vector<torch::Tensor> parameters;
    for(auto obj: _differentiable_components)
    {
        if(!obj->isOptimizable()) continue;
        auto obj_parameters = obj->getParameters();

        parameters.insert(parameters.end(), obj_parameters.begin(), obj_parameters.end());
    }

    return parameters;
}

void DiffPathtracingIntegrator::markOptimizable()
{
    for(auto obj: _differentiable_components)
    {
        obj->markOptimizable();
    }
}

torch::Tensor DiffPathtracingFunction::apply(const atcg::ref_ptr<DiffPathtracingIntegrator>& integrator,
                                             Dictionary& dict)
{
    torch::NoGradGuard no_grad;

    integrator->forwardPass(dict);
    auto target_detached = integrator->getHDR();

    torch::Tensor target = target_detached.clone().set_requires_grad(true);

    target.register_hook([&integrator](torch::Tensor adjoint_y) { integrator->backwardPass(adjoint_y); });

    return target;
}
}    // namespace atcg