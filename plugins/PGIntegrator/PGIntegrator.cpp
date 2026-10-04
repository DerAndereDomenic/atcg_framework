#include "PGIntegrator.h"

#ifdef ATCG_CUDA_BACKEND
    #include <Shape/Shape.h>
    #include <Shape/ShapeInstance.h>
    #include <Shape/MeshShape.h>
    #include <Emitter/MeshEmitter.h>
    #include <Scene/SceneAdapter.h>
    #include "PGKernels.h"
#endif

#include "vMF.h"

namespace atcg
{

PGIntegrator::PGIntegrator(const atcg::ref_ptr<RaytracingContext>& context, const Dictionary& dict)
    : Integrator(context, dict)
{
    initializePipeline(dict);
}

PGIntegrator::~PGIntegrator() {}

void PGIntegrator::initializePipeline(const Dictionary& dict)
{
    _pipeline->addTrianglesHitGroupShader("MeshShape", 0, {"./bin/MeshShape_ptx.ptx", "__closesthit__mesh"}, {});

    auto scene = dict.getValue<atcg::ref_ptr<Scene>>("scene");

    auto adapter    = SceneAdapter(_context, _pipeline, _sbt);
    _optix_scene    = adapter.apply(scene, dict.getValue<uint32_t>("width"), dict.getValue<uint32_t>("height"));
    auto scene_aabb = adapter.getSceneAABB();
    _scene_aabb.upload(&scene_aabb);

    const std::string ptx_raygen_filename = "./bin/PGIntegrator_ptx.ptx";
    OptixProgramGroup raygen_prog_group   = _pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__rg"});
    OptixProgramGroup miss_prog_group     = _pipeline->addMissShader({ptx_raygen_filename, "__miss__ms"});
    OptixProgramGroup occl_prog_group     = _pipeline->addMissShader({ptx_raygen_filename, "__miss__occlusion"});

    _raygen_index         = _sbt->addRaygenEntry(raygen_prog_group);
    _surface_miss_index   = _sbt->addMissEntry(miss_prog_group);
    _occlusion_miss_index = _sbt->addMissEntry(occl_prog_group);

    _pipeline->createPipeline();
    _sbt->createSBT();

    // Init distributions
    int T               = 64 * 64 * 64;
    _vMF_mixture_buffer = atcg::DeviceBuffer<vMFMixture<5>>(T);
    std::vector<vMFMixture<5>> vmf_mixtures(T);
    for(int i = 0; i < T; ++i)
    {
        for(int j = 0; j < 5; ++j)
        {
            vmf_mixtures[i].lobes[j].mu    = glm::vec3(1.0f, 0.0f, 0.0f);    // Initialize to some arbitrary direction
            vmf_mixtures[i].lobes[j].kappa = 00.0f;
            vmf_mixtures[i].weights[j]     = 1.0f / 5.0f;
        }
    }
    _vMF_mixture_buffer.upload(vmf_mixtures.data(), T);
}

void PGIntegrator::onImGuiRender()
{
#ifndef ATCG_HEADLESS
    ImGui::Begin("PGIntegrator");
    for(auto shape: _optix_scene->getShapes())
    {
        shape->onImGuiRender();
    }
    ImGui::End();

    ImGui::Begin("PGIntegrator Distributions");

    if(ImGui::Checkbox("Use Guiding", &use_guiding))
    {
        reset();
    }

    if(ImGui::Checkbox("Accumulate", &accumulate))
    {
        reset();
    }

    ImGui::End();
#endif
}

void PGIntegrator::reset()
{
    _frame_counter = 0;
    _optix_scene->getSensor()->getFilm()->clear();
    _optix_scene->getSensor()->markDirty();
}

void PGIntegrator::generateRays(Dictionary& in_out_dictionary)
{
    uint32_t width  = _optix_scene->getSensor()->getFilm()->getWidth();
    uint32_t height = _optix_scene->getSensor()->getFilm()->getHeight();

    torch::Tensor output_entities = torch::zeros({height, width}, atcg::TensorOptions::int32DeviceOptions());

    PGIntegratorParams params;

    params.sensor = _optix_scene->getSensor()->getVPtrTable();

    params.image_height = height;
    params.image_width  = width;
    params.handle       = _optix_scene->getIAS()->getTraversableHandle();
    params.scene_aabb   = _scene_aabb.get();

    params.entity_ids  = output_entities.numel() > 0 ? (int32_t*)output_entities.data_ptr() : nullptr;
    params.vmf_mixture = _vMF_mixture_buffer.get();
    params.num_vmfs    = _vMF_mixture_buffer.size();

    params.use_guiding = use_guiding;
    params.accumulate  = accumulate;

    params.frame_counter = _frame_counter++;

    params.num_emitters        = _optix_scene->getEmitterVPtrTables().size();
    params.emitters            = _optix_scene->getEmitterVPtrTables().get();
    auto environment_emitter   = _optix_scene->getEnvironmentEmitter();
    params.environment_emitter = environment_emitter ? environment_emitter->getVPtrTable() : nullptr;

    params.surface_trace_params = _pipeline->getRay(0, _surface_miss_index, false);

    params.occlusion_trace_params = _pipeline->getRay(0, _occlusion_miss_index, true);

    _launch_params.upload(&params);

    _pipeline->launch((CUdeviceptr)_launch_params.get(),
                      sizeof(PGIntegratorParams),
                      _sbt->getSBT(_raygen_index),
                      width,
                      height,
                      1);

    torch::Tensor output_tensor = _optix_scene->getSensor()->getFilm()->develop();
    in_out_dictionary.setValue("output", output_tensor);
    in_out_dictionary.setValue("entity_ids", output_entities);

    updateMixtures(_vMF_mixture_buffer);

    // TODO
    // Don't accumulate for now, just clear the film and mark the sensor as dirty
    if(!accumulate)
    {
        _optix_scene->getSensor()->getFilm()->clear();
        _optix_scene->getSensor()->markDirty();
    }
}
}    // namespace atcg