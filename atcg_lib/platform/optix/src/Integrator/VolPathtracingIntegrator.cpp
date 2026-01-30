#include <Integrator/VolPathtracingIntegrator.h>

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

#ifndef ATCG_HEADLESS
    #include <imgui.h>
#endif

// ! temp
#include <Film/HDRFilm.h>
#include <Sensor/PinholeCamera.h>

namespace atcg
{
VolPathtracingIntegrator::VolPathtracingIntegrator(const atcg::ref_ptr<RaytracingContext>& context,
                                                   const Dictionary& dict)
    : Integrator(context, dict)
{
    _scene = dict.getValue<atcg::ref_ptr<Scene>>("scene");

    atcg::Dictionary film_dict;
    film_dict.setValue<uint32_t>("width", dict.getValue<uint32_t>("width"));
    film_dict.setValue<uint32_t>("height", dict.getValue<uint32_t>("height"));
    atcg::ref_ptr<Film> film = atcg::make_ref<HDRFilm>(film_dict);

    atcg::Dictionary sensor_dict;
    sensor_dict.setValue("film", film);
    sensor_dict.setValue<atcg::ref_ptr<Camera>>("camera", _scene->getCamera());
    _sensor = atcg::make_ref<PinholeCamera>(sensor_dict);

    initializePipeline();
}

VolPathtracingIntegrator::~VolPathtracingIntegrator() {}

void VolPathtracingIntegrator::initializePipeline()
{
    _pipeline->addTrianglesHitGroupShader("MeshShape", 0, {"./bin/MeshShape_ptx.ptx", "__closesthit__mesh"}, {});

    const std::string ptx_raygen_filename = "./bin/VolPathtracingIntegrator_ptx.ptx";
    OptixProgramGroup raygen_prog_group   = _pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__rg"});
    OptixProgramGroup miss_prog_group     = _pipeline->addMissShader({ptx_raygen_filename, "__miss__ms"});
    OptixProgramGroup occl_prog_group     = _pipeline->addMissShader({ptx_raygen_filename, "__miss__occlusion"});

    _raygen_index         = _sbt->addRaygenEntry(raygen_prog_group);
    _surface_miss_index   = _sbt->addMissEntry(miss_prog_group);
    _occlusion_miss_index = _sbt->addMissEntry(occl_prog_group);

    _optix_scene = SceneAdapter(_context, _pipeline, _sbt).apply(_scene);

    _sensor->initializePipeline(_pipeline, _sbt);

    _pipeline->createPipeline();
    _sbt->createSBT();
}

void VolPathtracingIntegrator::onImGuiRender()
{
#ifndef ATCG_HEADLESS
    _panel.renderPanel(_optix_scene);
#endif
}

void VolPathtracingIntegrator::reset()
{
    _frame_counter = 0;
    _sensor->getFilm()->clear();
    _sensor->markDirty();
}

void VolPathtracingIntegrator::generateRays(Dictionary& in_out_dictionary)
{
    uint32_t width  = _sensor->getFilm()->getWidth();
    uint32_t height = _sensor->getFilm()->getHeight();

    torch::Tensor output_entities = torch::zeros({height, width}, atcg::TensorOptions::int32DeviceOptions());

    VolPathtracingParams params;

    params.sensor = _sensor->getVPtrTable();

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

    _launch_params.upload(&params);

    OPTIX_CHECK(optixLaunch(_pipeline->getPipeline(),
                            nullptr,
                            (CUdeviceptr)_launch_params.get(),
                            sizeof(VolPathtracingParams),
                            _sbt->getSBT(_raygen_index),
                            width,
                            height,
                            1));    // depth

    CUDA_SAFE_CALL(cudaStreamSynchronize(nullptr));

    torch::Tensor output_tensor = _sensor->getFilm()->develop();
    in_out_dictionary.setValue("output", output_tensor);
    in_out_dictionary.setValue("entity_ids", output_entities);
}
}    // namespace atcg