#include "NRCIntegrator.h"

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
#include <DataStructure/Timer.h>

#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <c10/cuda/CUDAGuard.h>

#ifndef ATCG_HEADLESS
    #include <imgui.h>
#endif

namespace atcg
{
NRCIntegrator::NRCIntegrator(const atcg::ref_ptr<RaytracingContext>& context, const Dictionary& dict)
    : Integrator(context, dict)
{
    initializePipeline(dict);

    _sample_generation_time = Statistic<float>("Sample Generation Time");
    _training_time          = Statistic<float>("Training Time");
    _render_time            = Statistic<float>("Render Time");
}

NRCIntegrator::~NRCIntegrator() {}

void NRCIntegrator::initializePipeline(const Dictionary& dict)
{
    _pipeline->addTrianglesHitGroupShader("MeshShape", 0, {"./bin/MeshShape_ptx.ptx", "__closesthit__mesh"}, {});

    auto scene = dict.getValue<atcg::ref_ptr<Scene>>("scene");

    uint32_t width  = dict.getValue<uint32_t>("width");
    uint32_t height = dict.getValue<uint32_t>("height");
    _optix_scene    = SceneAdapter(_context, _pipeline, _sbt).apply(scene, width, height);

    const std::string ptx_raygen_filename = "./bin/NRCIntegrator_ptx.ptx";
    OptixProgramGroup raygen_prog_group   = _pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__render"});
    OptixProgramGroup samplegen_prog_group =
        _pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__sample_generation"});
    OptixProgramGroup train_prog_group = _pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__train"});
    OptixProgramGroup miss_prog_group  = _pipeline->addMissShader({ptx_raygen_filename, "__miss__ms"});
    OptixProgramGroup occl_prog_group  = _pipeline->addMissShader({ptx_raygen_filename, "__miss__occlusion"});

    _raygen_render        = _sbt->addRaygenEntry(raygen_prog_group);
    _raygen_sample_gen    = _sbt->addRaygenEntry(samplegen_prog_group);
    _raygen_train         = _sbt->addRaygenEntry(train_prog_group);
    _surface_miss_index   = _sbt->addMissEntry(miss_prog_group);
    _occlusion_miss_index = _sbt->addMissEntry(occl_prog_group);

    _pipeline->createPipeline();
    _sbt->createSBT();

    _max_num_training_samples = width * height * 8;    // For now, the maximum possible number
    _training_samples         = atcg::DeviceBuffer<TrainingSample>(_max_num_training_samples);
    _training_sample_radiance = atcg::DeviceBuffer<SampledSpectrum>(width * height);
    int zero                  = 0;
    _training_samples_queue_index.upload(&zero);

    _weights = torch::empty({64 * 64 + 3 * 64 * 64 + 8 * 64}, atcg::TensorOptions::floatDeviceOptions());
    float a  = std::sqrt(6.0f / 128.0f);
    torch::nn::init::uniform_(_weights, -a, a);
    _weights = _weights.requires_grad_(true);
    _bias    = torch::zeros({64 + 64 + 64 + 64 + 8}, atcg::TensorOptions::floatDeviceOptions());
    _bias    = _bias.requires_grad_(true);

    _hash_grid    = atcg::HashGrid<half, 16, 2>(16, 512, (1 << 20));
    _hash_weights = _hash_grid.getWeights().to(torch::kFloat32).requires_grad_(true);

    _mlp       = atcg::MLP<3, 64, 64, 8>(_context, _weights.to(torch::kFloat16), _bias.to(torch::kFloat16));
    _optimizer = atcg::make_ref<torch::optim::Adam>(std::vector<torch::Tensor> {_weights, _bias, _hash_weights},
                                                    torch::optim::AdamOptions(1e-3));
}

void NRCIntegrator::onImGuiRender()
{
#ifndef ATCG_HEADLESS
    ImGui::Begin("NRCIntegrator");

    {
        std::stringstream ss;
        ss << _sample_generation_time;
        ImGui::Text(ss.str().c_str());
    }

    {
        std::stringstream ss;
        ss << _training_time;
        ImGui::Text(ss.str().c_str());
    }

    {
        std::stringstream ss;
        ss << _render_time;
        ImGui::Text(ss.str().c_str());
    }


    ImGui::Separator();

    for(auto shape: _optix_scene->getShapes())
    {
        shape->onImGuiRender();
    }
    ImGui::End();
#endif
}

void NRCIntegrator::reset()
{
    _frame_counter = 0;
    _optix_scene->getSensor()->getFilm()->clear();
    _optix_scene->getSensor()->markDirty();
}

void NRCIntegrator::generateRays(Dictionary& in_out_dictionary)
{
    {
        atcg::Timer timer;
        generateTrainingSamples();
        _sample_generation_time.addSample(timer.elapsedMillis());
    }
    {
        atcg::Timer timer;
        trainRadianceCache();
        _training_time.addSample(timer.elapsedMillis());
    }
    {
        atcg::Timer timer;
        renderWithRadianceCache(in_out_dictionary);
        _render_time.addSample(timer.elapsedMillis());
    }
}


void NRCIntegrator::generateTrainingSamples()
{
    uint32_t width  = _optix_scene->getSensor()->getFilm()->getWidth();
    uint32_t height = _optix_scene->getSensor()->getFilm()->getHeight();

    NRCParams params;

    params.sensor = _optix_scene->getSensor()->getVPtrTable();

    params.image_height = height;
    params.image_width  = width;
    params.handle       = _optix_scene->getIAS()->getTraversableHandle();

    params.entity_ids = nullptr;

    params.frame_counter = _frame_counter++;

    params.num_emitters        = _optix_scene->getEmitterVPtrTables().size();
    params.emitters            = _optix_scene->getEmitterVPtrTables().get();
    auto environment_emitter   = _optix_scene->getEnvironmentEmitter();
    params.environment_emitter = environment_emitter ? environment_emitter->getVPtrTable() : nullptr;

    params.surface_trace_params = _pipeline->getRay(0, _surface_miss_index, false);

    params.occlusion_trace_params = _pipeline->getRay(0, _occlusion_miss_index, true);

    int zero = 0;
    _training_samples_queue_index.upload(&zero);
    params.training_sample_radiance     = _training_sample_radiance.get();
    params.training_samples             = _training_samples.get();
    params.training_samples_queue_index = _training_samples_queue_index.get();
    params.max_training_samples         = _max_num_training_samples;

    _launch_params.upload(&params);

    _pipeline->launch((CUdeviceptr)_launch_params.get(),
                      sizeof(NRCParams),
                      _sbt->getSBT(_raygen_sample_gen),
                      width,
                      height,
                      1,
                      nullptr);
}

void NRCIntegrator::trainRadianceCache()
{
    _mlp.zeroGradients();
    _hash_grid.zeroGradients();
    NRCParams params;

    int num_samples = 0;
    _training_samples_queue_index.download(&num_samples);
    params.training_sample_radiance = _training_sample_radiance.get();
    params.training_samples         = _training_samples.get();
    params.max_training_samples     = (uint32_t)num_samples;
    params.mlp                      = _mlp.getDeviceMLP();
    params.hash_grid                = _hash_grid.getDeviceHashGrid();

    _launch_params.upload(&params);

    _pipeline->launch((CUdeviceptr)_launch_params.get(),
                      sizeof(NRCParams),
                      _sbt->getSBT(_raygen_train),
                      num_samples,
                      1,
                      1,
                      nullptr);

    _optimizer->zero_grad();

    _weights.mutable_grad()      = (_mlp.getWeightGradients().to(torch::kFloat32));
    _bias.mutable_grad()         = (_mlp.getBiasGradients().to(torch::kFloat32));
    _hash_weights.mutable_grad() = (_hash_grid.getGradWeights().to(torch::kFloat32));

    _weights.mutable_grad()      = torch::nan_to_num(_weights.grad(), 0.0f);
    _bias.mutable_grad()         = torch::nan_to_num(_bias.grad(), 0.0f);
    _hash_weights.mutable_grad() = torch::nan_to_num(_hash_weights.grad(), 0.0f);

    try
    {
        _optimizer->step();
    }
    catch(const std::exception& e)
    {
        ATCG_ERROR("Error during optimization step: {}", e.what());
    }

    _mlp.setWeights(_weights.to(torch::kFloat16));
    _mlp.setBias(_bias.to(torch::kFloat16));
    _hash_grid.setWeights(_hash_weights.to(torch::kFloat16));

    _mlp.uploadDeviceMLPData();
    _hash_grid.uploadDeviceHashGridData();
}

void NRCIntegrator::renderWithRadianceCache(Dictionary& in_out_dictionary)
{
    _optix_scene->getSensor()->getFilm()->clear();    // For now, don't accumulate

    uint32_t width  = _optix_scene->getSensor()->getFilm()->getWidth();
    uint32_t height = _optix_scene->getSensor()->getFilm()->getHeight();

    torch::Tensor output_entities = torch::zeros({height, width}, atcg::TensorOptions::int32DeviceOptions());

    NRCParams params;

    params.sensor = _optix_scene->getSensor()->getVPtrTable();

    params.image_height = height;
    params.image_width  = width;
    params.handle       = _optix_scene->getIAS()->getTraversableHandle();

    params.entity_ids = output_entities.numel() > 0 ? (int32_t*)output_entities.data_ptr() : nullptr;

    params.frame_counter = _frame_counter++;

    params.num_emitters        = _optix_scene->getEmitterVPtrTables().size();
    params.emitters            = _optix_scene->getEmitterVPtrTables().get();
    auto environment_emitter   = _optix_scene->getEnvironmentEmitter();
    params.environment_emitter = environment_emitter ? environment_emitter->getVPtrTable() : nullptr;

    params.surface_trace_params = _pipeline->getRay(0, _surface_miss_index, false);

    params.occlusion_trace_params = _pipeline->getRay(0, _occlusion_miss_index, true);

    params.mlp       = _mlp.getDeviceMLP();
    params.hash_grid = _hash_grid.getDeviceHashGrid();

    _launch_params.upload(&params);

    _pipeline->launch((CUdeviceptr)_launch_params.get(),
                      sizeof(NRCParams),
                      _sbt->getSBT(_raygen_render),
                      width,
                      height,
                      1,
                      nullptr);

    torch::Tensor output_tensor = _optix_scene->getSensor()->getFilm()->develop();
    in_out_dictionary.setValue("output", output_tensor);
    in_out_dictionary.setValue("entity_ids", output_entities);
}
}    // namespace atcg