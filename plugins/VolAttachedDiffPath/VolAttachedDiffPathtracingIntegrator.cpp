#include "VolAttachedDiffPathtracingIntegrator.h"

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

namespace atcg
{

torch::autograd::variable_list VolAttachedDiffPathNode::apply(torch::autograd::variable_list&& grads)
{
    auto adjoint_y = grads[0];
    Dictionary dict;
    dict.setValue("adjoint_y", adjoint_y);
    dict.setValue("current_sample", sample);
    dict.setValue("JL_buffer", JL);
    dict.setValue("rng_index", rng_index);

    integrator->_optix_scene->zeroGrad();
    integrator->_backwardTrace(dict);

    return integrator->_optix_scene->getParameterGradients();
}

void VolAttachedDiffPathNode::release_variables()
{
    sample.reset();
    JL.reset();
}

VolAttachedDiffPathtracingIntegrator::VolAttachedDiffPathtracingIntegrator(
    const atcg::ref_ptr<RaytracingContext>& context,
    const Dictionary& dict)
    : Integrator(context, dict)
{
    _pipeline = atcg::make_ref<RayTracingPipeline>(context, 2);
    initializePipeline(dict);
}

VolAttachedDiffPathtracingIntegrator::~VolAttachedDiffPathtracingIntegrator() {}

void VolAttachedDiffPathtracingIntegrator::initializePipeline(const Dictionary& dict)
{
    _pipeline->addTrianglesHitGroupShader("MeshShape", 0, {"./bin/MeshShape_ptx.ptx", "__closesthit__mesh"}, {});
    _pipeline->addTrianglesHitGroupShader("MeshShape",
                                          1,
                                          {"./bin/DualMeshShape_ptx.ptx", "__closesthit__dual_mesh"},
                                          {});

    auto scene = dict.getValue<atcg::ref_ptr<Scene>>("scene");

    const std::string ptx_raygen_filename = "./bin/VolAttachedDiffPathtracingIntegrator_ptx.ptx";
    OptixProgramGroup raygen_prog_group_forward =
        _pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__forward"});
    OptixProgramGroup miss_prog_group      = _pipeline->addMissShader({ptx_raygen_filename, "__miss__ms"});
    OptixProgramGroup dual_miss_prog_group = _pipeline->addMissShader({ptx_raygen_filename, "__miss__dual"});
    OptixProgramGroup occl_prog_group      = _pipeline->addMissShader({ptx_raygen_filename, "__miss__occlusion"});

    _raygen_index_forward = _sbt->addRaygenEntry(raygen_prog_group_forward);
    _surface_miss_index   = _sbt->addMissEntry(miss_prog_group);
    _dual_miss_index      = _sbt->addMissEntry(dual_miss_prog_group);
    _occlusion_miss_index = _sbt->addMissEntry(occl_prog_group);

    uint32_t width  = dict.getValue<uint32_t>("width");
    uint32_t height = dict.getValue<uint32_t>("height");

    _optix_scene = SceneAdapter(_context, _pipeline, _sbt).apply(scene, width, height);

    _dict.setValue("optix_scene", _optix_scene);

    TextureSpecification spec;
    spec.width       = width;
    spec.height      = height;
    spec.format      = TextureFormat::RGFLOAT;
    _last_JL_texture = Texture2D::create(spec);

    _pipeline->createPipeline();
    _sbt->createSBT();
}

void VolAttachedDiffPathtracingIntegrator::onImGuiRender()
{
    _panel.renderPanel(_optix_scene);

    ImGui::Begin("Derivative");

    if(_last_JL.defined())
    {
        ImGui::SliderInt("Derivative Channel", &_derivative_channel, 0, 17);

        auto normalize = [](torch::Tensor inp) -> torch::Tensor
        {
            auto min = torch::amin(inp);
            auto max = torch::amax(inp);
            auto y   = (inp - min) / (max - min);

            // auto y = torch::where(inp == 0.0f, 0.0f, torch::sigmoid(inp));

            return y;
        };

        auto pos_neg = [normalize](torch::Tensor inp) -> torch::Tensor
        {
            torch::Tensor pos = torch::relu(inp);

            torch::Tensor neg = torch::relu(-inp);

            torch::Tensor y = torch::concat({pos, neg}, /*dim=*/-1);

            return normalize(y);
        };

        auto slice =
            _last_JL.index({torch::indexing::Slice(), torch::indexing::Slice(), _derivative_channel}).unsqueeze(-1);
        auto data = pos_neg(slice);
        _last_JL_texture->setData(data);

        ImGui::Image((ImTextureID)_last_JL_texture->getID(),
                     ImVec2((int)(4 * _last_JL_texture->width()), (int)(4 * _last_JL_texture->height())),
                     ImVec2 {0, 1},
                     ImVec2 {1, 0});
        auto mini = torch::amin(slice).item<float>();
        auto maxi = torch::amax(slice).item<float>();
        auto mean = torch::mean(torch::abs(slice)).item<float>();

        ImGui::Text("Min: %.6f, Max: %.6f, Mean: %.6f", mini, maxi, mean);
    }

    ImGui::End();
}

void VolAttachedDiffPathtracingIntegrator::reset()
{
    _frame_counter = 0;
    _last_JL.zero_();
    _optix_scene->getSensor()->markDirty();
    _optix_scene->getSensor()->getFilm()->clear();
}

std::tuple<torch::Tensor, torch::Tensor>
VolAttachedDiffPathtracingIntegrator::_forwardTrace(Dictionary& in_out_dictionary)
{
    uint32_t rng_index = in_out_dictionary.getValueOr<uint32_t>("rng_index", _frame_counter++);

    uint32_t width  = _optix_scene->getSensor()->getFilm()->getWidth();
    uint32_t height = _optix_scene->getSensor()->getFilm()->getHeight();

    torch::Tensor current_sample = torch::zeros({height, width, 3}, atcg::TensorOptions::floatDeviceOptions());
    torch::Tensor current_JL     = torch::zeros({height, width, 3 * 6}, atcg::TensorOptions::floatDeviceOptions());

    VolAttachedDiffPathtracingParams params;

    params.sensor = _optix_scene->getSensor()->getVPtrTable();

    params.image_height = height;
    params.image_width  = width;
    params.handle       = _optix_scene->getIAS()->getTraversableHandle();

    params.current_sample = (glm::vec3*)current_sample.data_ptr();
    params.JL_buffer      = (atcg::mat6x3*)current_JL.data_ptr();

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
    _pipeline->launch((CUdeviceptr)_launch_params.get(),
                      sizeof(VolAttachedDiffPathtracingParams),
                      _sbt->getSBT(_raygen_index_forward),
                      width,
                      height,
                      1,
                      stream);

    return {current_sample, current_JL};
}

void VolAttachedDiffPathtracingIntegrator::_backwardTrace(Dictionary& in_out_dictionary)
{
    auto adjoint_y     = in_out_dictionary.getValue<torch::Tensor>("adjoint_y");
    auto sample        = in_out_dictionary.getValue<torch::Tensor>("current_sample");
    auto JL            = in_out_dictionary.getValue<torch::Tensor>("JL_buffer");
    uint32_t width     = adjoint_y.size(1);
    uint32_t height    = adjoint_y.size(0);
    uint32_t rng_index = in_out_dictionary.getValueOr<uint32_t>("rng_index", _frame_counter);

    VolAttachedDiffPathtracingParams params;

    params.sensor = _optix_scene->getSensor()->getVPtrTable();

    params.image_height = height;
    params.image_width  = width;
    params.handle       = _optix_scene->getIAS()->getTraversableHandle();


    params.current_sample = (glm::vec3*)sample.data_ptr();       // Input sample from forward pass
    params.adjoint_y      = (glm::vec3*)adjoint_y.data_ptr();    // Input 𝛿L
    params.JL_buffer      = (atcg::mat6x3*)JL.data_ptr();        // Input JL from forward pass

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
    _pipeline->launch((CUdeviceptr)_launch_params.get(),
                      sizeof(VolAttachedDiffPathtracingParams),
                      _sbt->getSBT(_raygen_index_forward),
                      width,
                      height,
                      1,
                      stream);
}

void VolAttachedDiffPathtracingIntegrator::generateRays(Dictionary& in_out_dictionary)
{
    const auto& parameters = _optix_scene->getParameters();

    bool is_executable = parameters.size() > 0 && torch::autograd::GradMode::is_enabled() &&
                         torch::autograd::any_variable_requires_grad(parameters);

    torch::Tensor result, JL;
    {
        torch::NoGradGuard no_grad;
        std::tie(result, JL) = _forwardTrace(in_out_dictionary);
    }

    static int counter = 0;
    counter++;
    if(_last_JL.defined() && _last_JL.sizes() == JL.sizes())
    {
        _last_JL = JL / counter + _last_JL * (counter - 1) / counter;
    }
    else
    {
        _last_JL = JL;
        counter  = 1;
    }
    _last_JL = torch::where(torch::isfinite(_last_JL), _last_JL, 0.0f);

    if(is_executable)
    {
        std::shared_ptr<VolAttachedDiffPathNode> node(new VolAttachedDiffPathNode(), torch::autograd::deleteNode);
        auto next_edges = torch::autograd::collect_next_edges(parameters);
        node->set_next_edges(std::move(next_edges));
        // node->clear_input_metadata();
        node->integrator = this;
        node->rng_index  = in_out_dictionary.getValueOr<uint32_t>("rng_index", 0);
        node->sample     = result;
        node->JL         = JL;

        torch::autograd::set_history(result, node);
    }

    in_out_dictionary.setValue("output_img", result);
}
}    // namespace atcg