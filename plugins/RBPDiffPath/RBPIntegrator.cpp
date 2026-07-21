#include "RBPIntegrator.h"

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
#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <Utils/Utils.h>

namespace atcg
{

torch::autograd::variable_list RBPNode::apply(torch::autograd::variable_list&& grads)
{
    auto adjoint_y = grads[0];
    Dictionary dict;
    dict.setValue("adjoint_y", adjoint_y);

    integrator->_optix_scene->zeroGrad();
    integrator->_backwardTrace(dict);

    return integrator->_optix_scene->getParameterGradients();
}

void RBPNode::release_variables() {}

RBPIntegrator::RBPIntegrator(const atcg::ref_ptr<RaytracingContext>& context, const Dictionary& dict)
    : Integrator(context, dict)
{
    initializePipeline(dict);
}

RBPIntegrator::~RBPIntegrator() {}

void RBPIntegrator::initializePipeline(const Dictionary& dict)
{
    _pipeline->addTrianglesHitGroupShader("MeshShape", 0, {"./bin/MeshShape_ptx.ptx", "__closesthit__mesh"}, {});

    auto scene = dict.getValue<atcg::ref_ptr<Scene>>("scene");

    const std::string ptx_raygen_filename = "./bin/RBPIntegrator_ptx.ptx";
    OptixProgramGroup raygen_prog_group_forward =
        _pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__forward"});
    OptixProgramGroup raygen_prog_group_backward =
        _pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__backward"});
    OptixProgramGroup miss_prog_group = _pipeline->addMissShader({ptx_raygen_filename, "__miss__ms"});
    OptixProgramGroup occl_prog_group = _pipeline->addMissShader({ptx_raygen_filename, "__miss__occlusion"});

    _raygen_index_forward  = _sbt->addRaygenEntry(raygen_prog_group_forward);
    _raygen_index_backward = _sbt->addRaygenEntry(raygen_prog_group_backward);
    _surface_miss_index    = _sbt->addMissEntry(miss_prog_group);
    _occlusion_miss_index  = _sbt->addMissEntry(occl_prog_group);

    _optix_scene = SceneAdapter(_context, _pipeline, _sbt)
                       .apply(scene, dict.getValue<uint32_t>("width"), dict.getValue<uint32_t>("height"));

    _dict.setValue("optix_scene", _optix_scene);

    _pipeline->createPipeline();
    _sbt->createSBT();
}

void RBPIntegrator::onImGuiRender()
{
    _panel.renderPanel(_optix_scene);
}

void RBPIntegrator::reset()
{
    _frame_counter = 0;
    _optix_scene->getSensor()->markDirty();
    _optix_scene->getSensor()->getFilm()->clear();
}

torch::Tensor RBPIntegrator::_forwardTrace(Dictionary& in_out_dictionary)
{
    uint32_t width     = _optix_scene->getSensor()->getFilm()->getWidth();
    uint32_t height    = _optix_scene->getSensor()->getFilm()->getHeight();
    uint32_t rng_index = _frame_counter++;

    torch::Tensor current_sample = torch::zeros({height, width, 3}, atcg::TensorOptions::floatDeviceOptions());

    RBPParams params;

    params.sensor = _optix_scene->getSensor()->getVPtrTable();

    params.image_height = height;
    params.image_width  = width;
    params.handle       = _optix_scene->getIAS()->getTraversableHandle();

    params.current_sample = (glm::vec3*)current_sample.data_ptr();

    params.rng_index = rng_index;

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

    auto stream = at::cuda::getCurrentCUDAStream();

    _pipeline->launch((CUdeviceptr)_launch_params.get(),
                      sizeof(RBPParams),
                      _sbt->getSBT(_raygen_index_forward),
                      width,
                      height,
                      1,
                      stream);

    return current_sample;
}

void RBPIntegrator::_backwardTrace(Dictionary& in_out_dictionary)
{
    auto adjoint_y     = in_out_dictionary.getValue<torch::Tensor>("adjoint_y");
    uint32_t width     = adjoint_y.size(1);
    uint32_t height    = adjoint_y.size(0);
    uint32_t rng_index = _frame_counter++;

    RBPParams params;

    params.sensor = _optix_scene->getSensor()->getVPtrTable();

    params.image_height = height;
    params.image_width  = width;
    params.handle       = _optix_scene->getIAS()->getTraversableHandle();

    params.adjoint_y = (glm::vec3*)adjoint_y.data_ptr();    // Input 𝛿L

    params.rng_index = rng_index;

    params.num_emitters        = _optix_scene->getEmitterVPtrTables().size();
    params.emitters            = _optix_scene->getEmitterVPtrTables().get();
    auto environment_emitter   = _optix_scene->getEnvironmentEmitter();
    params.environment_emitter = environment_emitter ? environment_emitter->getVPtrTable() : nullptr;

    params.surface_trace_params.rayFlags     = OPTIX_RAY_FLAG_NONE;
    params.surface_trace_params.SBToffset    = 0;
    params.surface_trace_params.SBTstride    = 2;
    params.surface_trace_params.missSBTIndex = _surface_miss_index;

    params.occlusion_trace_params.rayFlags  = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
    params.occlusion_trace_params.SBToffset = 0;
    params.occlusion_trace_params.SBTstride = 2;
    params.occlusion_trace_params.missSBTIndex = _occlusion_miss_index;

    _launch_params.upload(&params);

    auto stream = at::cuda::getCurrentCUDAStream();

    _pipeline->launch((CUdeviceptr)_launch_params.get(),
                      sizeof(RBPParams),
                      _sbt->getSBT(_raygen_index_backward),
                      width,
                      height,
                      1,
                      stream);
}

void RBPIntegrator::generateRays(Dictionary& in_out_dictionary)
{
    const auto& parameters = _optix_scene->getParameters();

    bool is_executable = parameters.size() > 0 && torch::autograd::GradMode::is_enabled() &&
                         torch::autograd::any_variable_requires_grad(parameters);

    torch::Tensor result;
    {
        torch::NoGradGuard no_grad;
        result = _forwardTrace(in_out_dictionary);
    }

    if(is_executable)
    {
        std::shared_ptr<RBPNode> node(new RBPNode(), torch::autograd::deleteNode);
        auto next_edges = torch::autograd::collect_next_edges(parameters);
        node->set_next_edges(std::move(next_edges));
        node->integrator = this;

        torch::autograd::set_history(result, node);
    }


    in_out_dictionary.setValue("output_img", result);
}
}    // namespace atcg