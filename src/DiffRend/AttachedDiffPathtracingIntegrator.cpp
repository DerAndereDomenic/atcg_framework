#include "AttachedDiffPathtracingIntegrator.h"

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
#include <Utils/Utils.h>
#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <c10/cuda/CUDAGuard.h>

#include <optix_stubs.h>

namespace atcg
{

torch::autograd::variable_list AttachedDiffPathNode::apply(torch::autograd::variable_list&& grads)
{
    auto adjoint_y = grads[0];
    Dictionary dict;
    dict.setValue("adjoint_y", adjoint_y);
    dict.setValue("current_sample", sample);
    dict.setValue("JL_buffer", JL);
    dict.setValue("rng_index", rng_index);
    dict.setValue("camera", camera);

    integrator->zeroGrad();
    integrator->_backwardTrace(dict);

    return integrator->getParameterGradients();
}

void AttachedDiffPathNode::release_variables()
{
    sample.reset();
    JL.reset();
}

AttachedDiffPathtracingIntegrator::AttachedDiffPathtracingIntegrator(const atcg::ref_ptr<RaytracingContext>& context,
                                                                     const Dictionary& dict)
    : DifferentiableIntegrator(context, dict)
{
    _pipeline = atcg::make_ref<RayTracingPipeline>(context, 2);
    initializePipeline(dict);
}

AttachedDiffPathtracingIntegrator::~AttachedDiffPathtracingIntegrator() {}

void AttachedDiffPathtracingIntegrator::initializePipeline(const Dictionary& dict)
{
    _pipeline->addTrianglesHitGroupShader("MeshShape", 0, {"./bin/MeshShape_ptx.ptx", "__closesthit__mesh"}, {});
    _pipeline->addTrianglesHitGroupShader("MeshShape",
                                          1,
                                          {"./bin/DualMeshShape_ptx.ptx", "__closesthit__dual_mesh"},
                                          {});

    auto scene = dict.getValue<atcg::ref_ptr<Scene>>("scene");

    const std::string ptx_raygen_filename = "./bin/AttachedDiffPathtracingIntegrator_ptx.ptx";
    OptixProgramGroup raygen_prog_group_forward =
        _pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__forward"});
    OptixProgramGroup miss_prog_group      = _pipeline->addMissShader({ptx_raygen_filename, "__miss__ms"});
    OptixProgramGroup dual_miss_prog_group = _pipeline->addMissShader({ptx_raygen_filename, "__miss__dual"});
    OptixProgramGroup occl_prog_group      = _pipeline->addMissShader({ptx_raygen_filename, "__miss__occlusion"});

    _raygen_index_forward = _sbt->addRaygenEntry(raygen_prog_group_forward);
    _surface_miss_index   = _sbt->addMissEntry(miss_prog_group);
    _dual_miss_index      = _sbt->addMissEntry(dual_miss_prog_group);
    _occlusion_miss_index = _sbt->addMissEntry(occl_prog_group);

    _optix_scene = SceneAdapter(_context, _pipeline, _sbt)
                       .apply(scene, dict.getValue<uint32_t>("width"), dict.getValue<uint32_t>("height"));

    _pipeline->createPipeline();
    _sbt->createSBT();

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

void AttachedDiffPathtracingIntegrator::onImGuiRender()
{
    _panel.renderPanel(_optix_scene);
}

void AttachedDiffPathtracingIntegrator::reset()
{
    // _frame_counter = 0;
}

std::tuple<torch::Tensor, torch::Tensor> AttachedDiffPathtracingIntegrator::_forwardTrace(Dictionary& in_out_dictionary)
{
    auto camera        = in_out_dictionary.getValue<atcg::ref_ptr<atcg::PerspectiveCamera>>("camera");
    uint32_t width     = in_out_dictionary.getValue<uint32_t>("width");
    uint32_t height    = in_out_dictionary.getValue<uint32_t>("height");
    uint32_t rng_index = in_out_dictionary.getValue<uint32_t>("rng_index");

    torch::Tensor current_sample = torch::zeros({height, width, 3}, atcg::TensorOptions::floatDeviceOptions());
    torch::Tensor current_JL     = torch::zeros({height, width, 3 * 4}, atcg::TensorOptions::floatDeviceOptions());

    AttachedDiffPathtracingParams params;

    glm::mat4 inv_camera_view = glm::inverse(camera->getView());
    memcpy(params.cam_eye, glm::value_ptr(inv_camera_view[3]), sizeof(glm::vec3));
    memcpy(params.U, glm::value_ptr(glm::normalize(inv_camera_view[0])), sizeof(glm::vec3));
    memcpy(params.V, glm::value_ptr(glm::normalize(inv_camera_view[1])), sizeof(glm::vec3));
    memcpy(params.W, glm::value_ptr(-glm::normalize(inv_camera_view[2])), sizeof(glm::vec3));
    params.fov_y = camera->getFOV();

    params.image_height = height;
    params.image_width  = width;
    params.handle       = _optix_scene->getIAS()->getTraversableHandle();

    params.current_sample = (glm::vec3*)current_sample.data_ptr();
    params.JL_buffer      = (glm::mat4x3*)current_JL.data_ptr();

    params.rng_index = rng_index;

    params.num_emitters        = _optix_scene->getEmitterVPtrTables().size();
    params.emitters            = _optix_scene->getEmitterVPtrTables().get();
    auto environment_emitter   = _optix_scene->getEnvironmentEmitter();
    params.environment_emitter = environment_emitter ? environment_emitter->getVPtrTable() : nullptr;

    params.surface_trace_params   = _pipeline->getRay(0, _surface_miss_index, false);
    params.occlusion_trace_params = _pipeline->getRay(0, _occlusion_miss_index, true);
    params.dual_trace_params      = _pipeline->getRay(1, _dual_miss_index, false);

    params.debug     = in_out_dictionary.getValueOr<bool>("debug", false);
    params.diff_mode = DiffMode::FORWARD;

    _launch_params.upload(&params);

    auto stream = at::cuda::getCurrentCUDAStream();
    OPTIX_CHECK(optixLaunch(_pipeline->getPipeline(),
                            stream,
                            (CUdeviceptr)_launch_params.get(),
                            sizeof(AttachedDiffPathtracingParams),
                            _sbt->getSBT(_raygen_index_forward),
                            width,
                            height,
                            1));    // depth

    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

    return {current_sample, current_JL};
}

void AttachedDiffPathtracingIntegrator::_backwardTrace(Dictionary& in_out_dictionary)
{
    auto camera        = in_out_dictionary.getValue<atcg::PerspectiveCamera*>("camera");
    auto adjoint_y     = in_out_dictionary.getValue<torch::Tensor>("adjoint_y");
    auto sample        = in_out_dictionary.getValue<torch::Tensor>("current_sample");
    auto JL            = in_out_dictionary.getValue<torch::Tensor>("JL_buffer");
    uint32_t width     = adjoint_y.size(1);
    uint32_t height    = adjoint_y.size(0);
    uint32_t rng_index = in_out_dictionary.getValue<uint32_t>("rng_index");

    AttachedDiffPathtracingParams params;

    glm::mat4 inv_camera_view = glm::inverse(camera->getView());
    memcpy(params.cam_eye, glm::value_ptr(inv_camera_view[3]), sizeof(glm::vec3));
    memcpy(params.U, glm::value_ptr(glm::normalize(inv_camera_view[0])), sizeof(glm::vec3));
    memcpy(params.V, glm::value_ptr(glm::normalize(inv_camera_view[1])), sizeof(glm::vec3));
    memcpy(params.W, glm::value_ptr(-glm::normalize(inv_camera_view[2])), sizeof(glm::vec3));
    params.fov_y = camera->getFOV();

    params.image_height = height;
    params.image_width  = width;
    params.handle       = _optix_scene->getIAS()->getTraversableHandle();


    params.current_sample = (glm::vec3*)sample.data_ptr();       // Input sample from forward pass
    params.adjoint_y      = (glm::vec3*)adjoint_y.data_ptr();    // Input 𝛿L
    params.JL_buffer      = (glm::mat4x3*)JL.data_ptr();         // Input JL from forward pass

    params.rng_index = rng_index;

    params.num_emitters        = _optix_scene->getEmitterVPtrTables().size();
    params.emitters            = _optix_scene->getEmitterVPtrTables().get();
    auto environment_emitter   = _optix_scene->getEnvironmentEmitter();
    params.environment_emitter = environment_emitter ? environment_emitter->getVPtrTable() : nullptr;

    params.surface_trace_params   = _pipeline->getRay(0, _surface_miss_index, false);
    params.occlusion_trace_params = _pipeline->getRay(0, _occlusion_miss_index, true);
    params.dual_trace_params      = _pipeline->getRay(1, _dual_miss_index, false);

    params.debug     = in_out_dictionary.getValueOr<bool>("debug", false);
    params.diff_mode = DiffMode::BACKWARD;

    _launch_params.upload(&params);

    auto stream = at::cuda::getCurrentCUDAStream();
    OPTIX_CHECK(optixLaunch(_pipeline->getPipeline(),
                            stream,
                            (CUdeviceptr)_launch_params.get(),
                            sizeof(AttachedDiffPathtracingParams),
                            _sbt->getSBT(_raygen_index_forward),
                            width,
                            height,
                            1));    // depth

    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
}

torch::Tensor AttachedDiffPathtracingIntegrator::sample(Dictionary& in_out_dictionary)
{
    const auto& parameters = getParameters();

    bool is_executable = parameters.size() > 0 && torch::autograd::GradMode::is_enabled() &&
                         torch::autograd::any_variable_requires_grad(parameters);

    torch::Tensor result, JL;
    {
        torch::NoGradGuard no_grad;
        std::tie(result, JL) = _forwardTrace(in_out_dictionary);
    }

    if(is_executable)
    {
        std::shared_ptr<AttachedDiffPathNode> node(new AttachedDiffPathNode(), torch::autograd::deleteNode);
        auto next_edges = torch::autograd::collect_next_edges(parameters);
        node->set_next_edges(std::move(next_edges));
        // node->clear_input_metadata();
        node->integrator = this;
        node->rng_index  = in_out_dictionary.getValueOr<uint32_t>("rng_index", 0);
        node->camera     = in_out_dictionary.getValue<atcg::ref_ptr<atcg::PerspectiveCamera>>("camera").get();
        node->sample     = result;
        node->JL         = JL;

        torch::autograd::set_history(result, node);
    }


    return result;
}

std::vector<torch::Tensor> AttachedDiffPathtracingIntegrator::getParameters() const
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

std::vector<torch::Tensor> AttachedDiffPathtracingIntegrator::getParameterGradients() const
{
    std::vector<torch::Tensor> gradients;
    for(auto obj: _differentiable_components)
    {
        if(!obj->isOptimizable()) continue;
        auto obj_gradients = obj->getParameterGradients();

        gradients.insert(gradients.end(), obj_gradients.begin(), obj_gradients.end());
    }

    return gradients;
}

void AttachedDiffPathtracingIntegrator::zeroGrad()
{
    for(auto obj: _differentiable_components)
    {
        if(!obj->isOptimizable()) continue;
        obj->zeroGrad();
    }
}

void AttachedDiffPathtracingIntegrator::clampParameters()
{
    for(auto obj: _differentiable_components)
    {
        if(!obj->isOptimizable()) continue;
        obj->clampParameters();
    }
}

void AttachedDiffPathtracingIntegrator::markOptimizable()
{
    for(auto obj: _differentiable_components)
    {
        obj->markOptimizable();
    }
}
}    // namespace atcg