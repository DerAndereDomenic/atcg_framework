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

void DiffPathtracingIntegrator::reset()
{
    _frame_counter = 0;
}

torch::Tensor DiffPathtracingIntegrator::getHDR() const
{
    return _accumulation_buffer.clone();
}

void DiffPathtracingIntegrator::forwardPass(Dictionary& in_out_dictionary)
{
    auto camera     = in_out_dictionary.getValue<atcg::ref_ptr<atcg::PerspectiveCamera>>("camera");
    auto output_img = in_out_dictionary.getValue<torch::Tensor>("output_img");
    auto entity_ids = in_out_dictionary.getValueOr<torch::Tensor>("entity_ids", torch::empty({0}));

    if(_accumulation_buffer.numel() == 0 || _frame_counter == 0 || _accumulation_buffer.size(0) != output_img.size(0) ||
       _accumulation_buffer.size(1) != output_img.size(1))
    {
        _accumulation_buffer =
            torch::zeros({output_img.size(0), output_img.size(1), 3}, atcg::TensorOptions::floatDeviceOptions());

        _adjoint_x =
            torch::zeros({output_img.size(0), output_img.size(1), 3}, atcg::TensorOptions::floatDeviceOptions());

        _adjoint_y =
            torch::zeros({output_img.size(0), output_img.size(1), 3}, atcg::TensorOptions::floatDeviceOptions());
    }

    DiffPathtracingParams params;

    glm::mat4 inv_camera_view = glm::inverse(camera->getView());
    memcpy(params.cam_eye, glm::value_ptr(inv_camera_view[3]), sizeof(glm::vec3));
    memcpy(params.U, glm::value_ptr(glm::normalize(inv_camera_view[0])), sizeof(glm::vec3));
    memcpy(params.V, glm::value_ptr(glm::normalize(inv_camera_view[1])), sizeof(glm::vec3));
    memcpy(params.W, glm::value_ptr(-glm::normalize(inv_camera_view[2])), sizeof(glm::vec3));
    params.fov_y = camera->getFOV();

    params.image_height = output_img.size(0);
    params.image_width  = output_img.size(1);
    params.handle       = _optix_scene->getIAS()->getTraversableHandle();

    params.entity_ids = entity_ids.numel() > 0 ? (int32_t*)entity_ids.data_ptr() : nullptr;

    params.accumulation_buffer = (glm::vec3*)_accumulation_buffer.data_ptr();

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
                            output_img.size(1),
                            output_img.size(0),
                            1));    // depth

    CUDA_SAFE_CALL(cudaStreamSynchronize(nullptr));
}

void DiffPathtracingIntegrator::backwardPass(Dictionary& in_out_dictionary)
{
    auto camera     = in_out_dictionary.getValue<atcg::ref_ptr<atcg::PerspectiveCamera>>("camera");
    auto output_img = in_out_dictionary.getValue<torch::Tensor>("output_img");
    auto entity_ids = in_out_dictionary.getValueOr<torch::Tensor>("entity_ids", torch::empty({0}));

    // Dont reset can happen in backward pass
    // if(_accumulation_buffer.numel() == 0 || _frame_counter == 0 || _accumulation_buffer.size(0) != output_img.size(0)
    // ||
    //    _accumulation_buffer.size(1) != output_img.size(1))
    // {
    //     _accumulation_buffer =
    //         torch::zeros({output_img.size(0), output_img.size(1), 3}, atcg::TensorOptions::floatDeviceOptions());

    //     _adjoint_x =
    //         torch::zeros({output_img.size(0), output_img.size(1), 3}, atcg::TensorOptions::floatDeviceOptions());

    //     _adjoint_y =
    //         torch::zeros({output_img.size(0), output_img.size(1), 3}, atcg::TensorOptions::floatDeviceOptions());
    // }

    DiffPathtracingParams params;

    glm::mat4 inv_camera_view = glm::inverse(camera->getView());
    memcpy(params.cam_eye, glm::value_ptr(inv_camera_view[3]), sizeof(glm::vec3));
    memcpy(params.U, glm::value_ptr(glm::normalize(inv_camera_view[0])), sizeof(glm::vec3));
    memcpy(params.V, glm::value_ptr(glm::normalize(inv_camera_view[1])), sizeof(glm::vec3));
    memcpy(params.W, glm::value_ptr(-glm::normalize(inv_camera_view[2])), sizeof(glm::vec3));
    params.fov_y = camera->getFOV();

    params.image_height = output_img.size(0);
    params.image_width  = output_img.size(1);
    params.handle       = _optix_scene->getIAS()->getTraversableHandle();

    params.entity_ids = entity_ids.numel() > 0 ? (int32_t*)entity_ids.data_ptr() : nullptr;

    params.accumulation_buffer = (glm::vec3*)_accumulation_buffer.data_ptr();    // Input L
    params.adjoint_y           = (glm::vec3*)_adjoint_y.data_ptr();              // Input 𝛿L

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
                            output_img.size(1),
                            output_img.size(0),
                            1));    // depth

    CUDA_SAFE_CALL(cudaStreamSynchronize(nullptr));
}

void DiffPathtracingIntegrator::generateRays(Dictionary& in_out_dictionary)
{
    auto output_img = in_out_dictionary.getValue<torch::Tensor>("output_img");
    if(!_optimize)
    {
        forwardPass(in_out_dictionary);

        // Perform tonemapping here for output display:
        torch::Tensor tonemapped = torch::pow(1.0f - torch::exp(-_accumulation_buffer), 1.0 / 2.4f);

        tonemapped.clamp_(0.0f, 1.0f);
        output_img.fill_(255);
        output_img.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
                              (tonemapped * 255.0f).to(torch::kUInt8));
    }
    else
    {
        torch::Tensor target       = in_out_dictionary.getValue<torch::Tensor>("target");
        const uint32_t num_samples = 16;

        // Forward Pass
        for(int i = 0; i < num_samples; ++i)
        {
            forwardPass(in_out_dictionary);
        }

        // Backward pass
        _adjoint_y = 2.0f * (_accumulation_buffer - target);    // W is delta peak in our case

        reset();    // Use same random numbers

        for(auto diff: _differentiable_components)
        {
            diff->zero_grad();
        }

        for(int i = 0; i < num_samples; ++i)
        {
            backwardPass(in_out_dictionary);
        }

        // TODO: Update

        float lr = 0.1f;
        for(auto diff: _differentiable_components)
        {
            diff->update(lr);
        }

        // torch::Tensor _grad = torch::mean(_adjoint_x, at::IntArrayRef {0, 1}).cpu();    // Just reduce mean?

        // glm::vec3 grad = glm::vec3(_grad[0].item<float>(), _grad[1].item<float>(), _grad[2].item<float>());

        // const float lr = 1e0f;

        // ATCG_DEBUG(grad);

        // _albedo -= lr * grad;
        // _albedo = glm::clamp(_albedo, glm::vec3(0), glm::vec3(1));

        torch::Tensor tonemapped = torch::pow(1.0f - torch::exp(-torch::abs(_accumulation_buffer)), 1.0 / 2.4f);

        tonemapped.clamp_(0.0f, 1.0f);
        output_img.fill_(255);
        output_img.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
                              (tonemapped * 255.0f).to(torch::kUInt8));
    }
}
}    // namespace atcg