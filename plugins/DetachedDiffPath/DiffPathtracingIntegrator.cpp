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

torch::autograd::variable_list DiffPathNode::apply(torch::autograd::variable_list&& grads)
{
    auto adjoint_y = grads[0];
    Dictionary dict;
    dict.setValue("adjoint_y", adjoint_y);
    dict.setValue("current_sample", sample);
    dict.setValue("rng_index", rng_index);

    integrator->_optix_scene->zeroGrad();
    integrator->backwardTrace(dict);

    return integrator->_optix_scene->getParameterGradients();
}

void DiffPathNode::release_variables()
{
    sample.reset();
}

DiffPathtracingIntegrator::DiffPathtracingIntegrator(const atcg::ref_ptr<RaytracingContext>& context,
                                                     const Dictionary& dict)
    : Integrator(context, dict)
{
    initializePipeline(dict);
}

DiffPathtracingIntegrator::~DiffPathtracingIntegrator() {}

void DiffPathtracingIntegrator::initializePipeline(const Dictionary& dict)
{
    _pipeline->addTrianglesHitGroupShader("MeshShape", 0, {"./bin/MeshShape_ptx.ptx", "__closesthit__mesh"}, {});

    auto scene = dict.getValue<atcg::ref_ptr<Scene>>("scene");

    const std::string ptx_raygen_filename = "./bin/DiffPathtracingIntegrator_ptx.ptx";
    OptixProgramGroup raygen_prog_group_forward =
        _pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__forward"});
    OptixProgramGroup miss_prog_group = _pipeline->addMissShader({ptx_raygen_filename, "__miss__ms"});
    OptixProgramGroup occl_prog_group = _pipeline->addMissShader({ptx_raygen_filename, "__miss__occlusion"});

    _raygen_index_forward = _sbt->addRaygenEntry(raygen_prog_group_forward);
    _surface_miss_index   = _sbt->addMissEntry(miss_prog_group);
    _occlusion_miss_index = _sbt->addMissEntry(occl_prog_group);


    uint32_t width  = dict.getValue<uint32_t>("width");
    uint32_t height = dict.getValue<uint32_t>("height");
    _optix_scene    = SceneAdapter(_context, _pipeline, _sbt).apply(scene, width, height);

    _dict.setValue("optix_scene", _optix_scene);

    uint32_t num_aovs = dict.getValueOr<uint32_t>("num_aovs", 0);

    if(num_aovs > 0)
    {
        _aov_buffers.resize(num_aovs);
        _aov_buffer_pointers = atcg::DeviceBuffer<float*>(num_aovs);
        std::vector<float*> aov_buffer_pointers(num_aovs);
        for(uint32_t i = 0; i < num_aovs; ++i)
        {
            _aov_buffers[i]        = torch::zeros({height, width}, atcg::TensorOptions::floatDeviceOptions());
            aov_buffer_pointers[i] = reinterpret_cast<float*>(_aov_buffers[i].data_ptr());
        }
        _aov_buffer_pointers.upload(aov_buffer_pointers.data());
    }

    _pipeline->createPipeline();
    _sbt->createSBT();
}

void DiffPathtracingIntegrator::onImGuiRender()
{
    _panel.renderPanel(_optix_scene);
}

void DiffPathtracingIntegrator::reset()
{
    _frame_counter = 0;
    _optix_scene->getSensor()->markDirty();
    _optix_scene->getSensor()->getFilm()->clear();
}

torch::Tensor DiffPathtracingIntegrator::forwardTrace(Dictionary& in_out_dictionary)
{
    uint32_t width     = _optix_scene->getSensor()->getFilm()->getWidth();
    uint32_t height    = _optix_scene->getSensor()->getFilm()->getHeight();
    uint32_t rng_index = in_out_dictionary.getValueOr<uint32_t>("rng_index", _frame_counter++);

    torch::Tensor current_sample = torch::zeros({height, width, 3}, atcg::TensorOptions::floatDeviceOptions());

    DiffPathtracingParams params;

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

    params.diff_mode = DiffMode::FORWARD;

    _launch_params.upload(&params);

    auto stream = at::cuda::getCurrentCUDAStream();

    _pipeline->launch((CUdeviceptr)_launch_params.get(),
                      sizeof(DiffPathtracingParams),
                      _sbt->getSBT(_raygen_index_forward),
                      width,
                      height,
                      1,
                      stream);

    return current_sample;
}

void DiffPathtracingIntegrator::backwardTrace(Dictionary& in_out_dictionary)
{
    for(int i = 0; i < _aov_buffers.size(); ++i)
    {
        _aov_buffers[i].zero_();
    }

    auto adjoint_y     = in_out_dictionary.getValue<torch::Tensor>("adjoint_y");
    auto sample        = in_out_dictionary.getValue<torch::Tensor>("current_sample");
    uint32_t width     = adjoint_y.size(1);
    uint32_t height    = adjoint_y.size(0);
    uint32_t rng_index = in_out_dictionary.getValue<uint32_t>("rng_index");

    DiffPathtracingParams params;

    params.sensor = _optix_scene->getSensor()->getVPtrTable();

    params.image_height = height;
    params.image_width  = width;
    params.handle       = _optix_scene->getIAS()->getTraversableHandle();

    params.current_sample = (glm::vec3*)sample.data_ptr();       // Input L
    params.adjoint_y      = (glm::vec3*)adjoint_y.data_ptr();    // Input 𝛿L

    params.aov_buffers = _aov_buffer_pointers.get();
    params.num_aovs    = _aov_buffers.size();

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

    params.diff_mode = DiffMode::BACKWARD;

    _launch_params.upload(&params);

    auto stream = at::cuda::getCurrentCUDAStream();

    _pipeline->launch((CUdeviceptr)_launch_params.get(),
                      sizeof(DiffPathtracingParams),
                      _sbt->getSBT(_raygen_index_forward),
                      width,
                      height,
                      1,
                      stream);
}

void DiffPathtracingIntegrator::generateRays(Dictionary& in_out_dictionary)
{
    const auto& parameters = _optix_scene->getParameters();

    bool is_executable = parameters.size() > 0 && torch::autograd::GradMode::is_enabled() &&
                         torch::autograd::any_variable_requires_grad(parameters);

    torch::Tensor result;
    {
        torch::NoGradGuard no_grad;
        result = this->forwardTrace(in_out_dictionary);
    }

    if(is_executable)
    {
        std::shared_ptr<DiffPathNode> node(new DiffPathNode(), torch::autograd::deleteNode);
        auto next_edges = torch::autograd::collect_next_edges(parameters);
        node->set_next_edges(std::move(next_edges));
        node->integrator = this;
        node->rng_index  = in_out_dictionary.getValueOr<uint32_t>("rng_index", 0);
        node->sample     = result;

        torch::autograd::set_history(result, node);
    }

    in_out_dictionary.setValue("output_img", result);
}
}    // namespace atcg