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
    const std::string ptx_raygen_filename = "./bin/DiffPathtracingIntegrator_ptx.ptx";
    OptixProgramGroup raygen_prog_group   = pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__rg"});
    OptixProgramGroup miss_prog_group     = pipeline->addMissShader({ptx_raygen_filename, "__miss__ms"});
    OptixProgramGroup occl_prog_group     = pipeline->addMissShader({ptx_raygen_filename, "__miss__occlusion"});

    _raygen_index         = sbt->addRaygenEntry(raygen_prog_group);
    _surface_miss_index   = sbt->addMissEntry(miss_prog_group);
    _occlusion_miss_index = sbt->addMissEntry(occl_prog_group);

    _pipeline = pipeline;
    _sbt      = sbt;

    _optix_scene = SceneAdapter(_context, pipeline, sbt).apply(_scene);
}

void DiffPathtracingIntegrator::reset()
{
    _frame_counter = 0;
}

void DiffPathtracingIntegrator::registerTarget()
{
    _target = _hdr.clone();
}

void DiffPathtracingIntegrator::_generateSample(Dictionary& in_out_dictionary)
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

        _hdr = torch::zeros({output_img.size(0), output_img.size(1), 3}, atcg::TensorOptions::floatDeviceOptions());
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
    params.adjoint_x           = (glm::vec3*)_adjoint_x.data_ptr();
    params.albedo_x            = _albedo.x;
    params.albedo_y            = _albedo.y;
    params.albedo_z            = _albedo.z;

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
                            _sbt->getSBT(_raygen_index),
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
        _generateSample(in_out_dictionary);
        _hdr = (_hdr * (_frame_counter - 1) + _accumulation_buffer) / _frame_counter;

        // Perform tonemapping here for output display:
        torch::Tensor tonemapped = torch::pow(1.0f - torch::exp(-_hdr), 1.0 / 2.4f);

        tonemapped.clamp_(0.0f, 1.0f);
        output_img.fill_(255);
        output_img.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
                              (tonemapped * 255.0f).to(torch::kUInt8));
    }
    else
    {
        const uint32_t num_samples = 16;

        // Forward Pass
        torch::Tensor x  = torch::zeros_like(_accumulation_buffer);
        torch::Tensor dx = torch::zeros_like(_adjoint_x);
        for(int i = 0; i < 16; ++i)
        {
            _generateSample(in_out_dictionary);

            x += _accumulation_buffer;
            dx += _adjoint_x;
        }

        x /= (float)num_samples;
        dx /= (float)num_samples;

        // Backward pass
        torch::Tensor delta_y = 2.0f * (x - _target);
        torch::Tensor Ae      = delta_y;    // W is delta peak in our case

        dx = Ae * dx;    // Gradient image

        torch::Tensor _grad = torch::mean(dx, at::IntArrayRef {0, 1}).cpu();    // Just reduce mean?

        glm::vec3 grad = glm::vec3(_grad[0].item<float>(), _grad[1].item<float>(), _grad[2].item<float>());

        const float lr = 1e-0f;

        ATCG_DEBUG(grad);

        _albedo -= lr * grad;
        _albedo = glm::clamp(_albedo, glm::vec3(0), glm::vec3(1));

        torch::Tensor tonemapped = torch::pow(1.0f - torch::exp(-torch::abs(x)), 1.0 / 2.4f);

        tonemapped.clamp_(0.0f, 1.0f);
        output_img.fill_(255);
        output_img.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
                              (tonemapped * 255.0f).to(torch::kUInt8));
    }
}
}    // namespace atcg