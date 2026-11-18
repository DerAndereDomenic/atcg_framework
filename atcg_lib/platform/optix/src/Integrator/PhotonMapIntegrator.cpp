#include <Integrator/PhotonMapIntegrator.h>

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
#include <DataStructure/cuBQL.h>

#include <optix_stubs.h>

#ifndef ATCG_HEADLESS
    #include <imgui.h>
#endif

namespace atcg
{
PhotonMapIntegrator::PhotonMapIntegrator(const atcg::ref_ptr<RaytracingContext>& context, const Dictionary& dict)
    : Integrator(context, dict)
{
    initializePipeline(dict);

    _reduction_factor = dict.getValueOr<float>("reduction_factor", 0.7f);
    _gather_radius    = dict.getValueOr<float>("gather_radius", 0.25f);
    _max_photon_depth = dict.getValueOr<uint32_t>("max_photon_depth", 16);
    _max_camera_depth = dict.getValueOr<uint32_t>("max_camera_depth", 16);
    _max_num_photons  = dict.getValueOr<uint32_t>("max_num_photons", 1000000);
}

PhotonMapIntegrator::~PhotonMapIntegrator() {}

void PhotonMapIntegrator::initializePipeline(const Dictionary& dict)
{
    _pipeline->addTrianglesHitGroupShader("MeshShape", 0, {"MeshShape_ptx", "__closesthit__mesh"}, {});

    auto scene = dict.getValue<atcg::ref_ptr<Scene>>("scene");

    _optix_scene = SceneAdapter(_context, _pipeline, _sbt)
                       .apply(scene, dict.getValue<uint32_t>("width"), dict.getValue<uint32_t>("height"));

    const std::string ptx_raygen_filename = "PhotonMapIntegrator_ptx";
    OptixProgramGroup raygen_prog_group   = _pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__rg"});
    OptixProgramGroup raygen_sample_photons_prog_group =
        _pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__sample_photons"});
    OptixProgramGroup miss_prog_group = _pipeline->addMissShader({ptx_raygen_filename, "__miss__ms"});

    _raygen_index                = _sbt->addRaygenEntry(raygen_prog_group);
    _raygen_sample_photons_index = _sbt->addRaygenEntry(raygen_sample_photons_prog_group);
    _surface_miss_index          = _sbt->addMissEntry(miss_prog_group);

    _pipeline->createPipeline();
    _sbt->createSBT();

    _photon_data   = DeviceBuffer<PhotonMapData>(_max_num_photons);
    _photon_bounds = DeviceBuffer<cuBQL::box3f>(_max_num_photons);
    _photon_gather_data =
        DeviceBuffer<PhotonGatherData>(dict.getValue<uint32_t>("width") * dict.getValue<uint32_t>("height"));
}

void PhotonMapIntegrator::onImGuiRender()
{
#ifndef ATCG_HEADLESS
    ImGui::Begin("PhotonMapIntegrator");
    for(auto shape: _optix_scene->getShapes())
    {
        shape->onImGuiRender();
    }
    ImGui::End();
#endif
}

void PhotonMapIntegrator::reset()
{
    _frame_counter = 0;
    _optix_scene->getSensor()->getFilm()->clear();
    _optix_scene->getSensor()->markDirty();
}

void PhotonMapIntegrator::generateNewPhotonMap()
{
    tracePhotons();
    buildPhotonMap();
}

void PhotonMapIntegrator::tracePhotons()
{
    PhotonMapParams params;

    params.handle = _optix_scene->getIAS()->getTraversableHandle();

    params.frame_counter = _frame_counter;

    params.num_emitters        = _optix_scene->getEmitterVPtrTables().size();
    params.emitters            = _optix_scene->getEmitterVPtrTables().get();
    auto environment_emitter   = _optix_scene->getEnvironmentEmitter();
    params.environment_emitter = environment_emitter ? environment_emitter->getVPtrTable() : nullptr;

    params.surface_trace_params = _pipeline->getRay(0, _surface_miss_index, false);

    params.photon_data   = _photon_data.get();
    params.photon_bounds = _photon_bounds.get();
    int zero             = 0;
    _photon_index.upload(&zero);
    params.photon_index       = _photon_index.get();
    params.max_num_photons    = _max_num_photons;
    params.photons_per_launch = ((_max_num_photons) / (_max_photon_depth));
    params.reduction_factor   = _reduction_factor;
    params.gather_radius_sq   = _gather_radius * _gather_radius;
    params.max_photon_depth   = _max_photon_depth;
    params.max_camera_depth   = _max_camera_depth;

    _launch_params.upload(&params);

    OPTIX_CHECK(optixLaunch(_pipeline->getPipeline(),
                            nullptr,
                            (CUdeviceptr)_launch_params.get(),
                            sizeof(PhotonMapParams),
                            _sbt->getSBT(_raygen_sample_photons_index),
                            params.photons_per_launch,
                            1,
                            1));    // depth

    CUDA_SAFE_CALL(cudaStreamSynchronize(nullptr));

    // int generated_photons = 0;
    // _photon_index.download(&generated_photons);
    // ATCG_DEBUG("Generated photons: {}", generated_photons);
}

void PhotonMapIntegrator::buildPhotonMap()
{
    int num_photons;
    _photon_index.download(&num_photons);
    atcg::build3fBVH(_photon_bvh, _photon_bounds.get(), num_photons);
    _device_photon_bvh.upload(&_photon_bvh);
}


void PhotonMapIntegrator::generateRays(Dictionary& in_out_dictionary)
{
    // Generate a new PM for each frame for PPM
    generateNewPhotonMap();

    uint32_t width  = _optix_scene->getSensor()->getFilm()->getWidth();
    uint32_t height = _optix_scene->getSensor()->getFilm()->getHeight();

    torch::Tensor output_entities = torch::zeros({height, width}, atcg::TensorOptions::int32DeviceOptions());

    PhotonMapParams params;

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

    params.photon_bounds      = _photon_bounds.get();
    params.photon_bvh         = _device_photon_bvh.get();
    params.photon_data        = _photon_data.get();
    params.photon_index       = _photon_index.get();
    params.photon_gather_data = _photon_gather_data.get();

    params.photons_per_launch = ((_max_num_photons) / (_max_photon_depth));
    params.reduction_factor   = _reduction_factor;
    params.gather_radius_sq   = _gather_radius * _gather_radius;
    params.max_photon_depth   = _max_photon_depth;
    params.max_camera_depth   = _max_camera_depth;

    _launch_params.upload(&params);

    OPTIX_CHECK(optixLaunch(_pipeline->getPipeline(),
                            nullptr,
                            (CUdeviceptr)_launch_params.get(),
                            sizeof(PhotonMapParams),
                            _sbt->getSBT(_raygen_index),
                            width,
                            height,
                            1));    // depth

    CUDA_SAFE_CALL(cudaStreamSynchronize(nullptr));

    torch::Tensor output_tensor = _optix_scene->getSensor()->getFilm()->develop();
    in_out_dictionary.setValue("output", output_tensor);
    in_out_dictionary.setValue("entity_ids", output_entities);

    atcg::free3fBVH(_photon_bvh);
}

void PhotonMapIntegrator::registerIntegrator(IntegratorRegistry::Registry* registry)
{
    ATCG_REGISTER_INTEGRATOR(registry, PhotonMapIntegrator::getType(), PhotonMapIntegrator);
}


bool IntegratorSettings<PhotonMapIntegrator>::renderSettings()
{
    bool updated = false;
#ifndef ATCG_HEADLESS
    updated = ImGui::DragInt("Max Photon Depth", (int*)&_max_photon_depth, 1.0f, 1, INT_MAX) || updated;
    updated = ImGui::DragInt("Max Camera Depth", (int*)&_max_camera_depth, 1.0f, 1, INT_MAX) || updated;
    updated = ImGui::DragInt("Max Number of Photons", (int*)&_max_num_photons, 1.0f, 1, INT_MAX) || updated;
    updated = ImGui::DragFloat("Gather Radius", &_gather_radius, 0.1f, 0.1f, 100.0f) || updated;
    updated = ImGui::DragFloat("Reduction Factor", &_reduction_factor, 0.1f, 0.1f, 1.0f) || updated;
#endif
    return updated;
}

atcg::Dictionary IntegratorSettings<PhotonMapIntegrator>::toDictionary() const
{
    atcg::Dictionary dict;
    dict.setValue("reduction_factor", _reduction_factor);
    dict.setValue("gather_radius_sq", _gather_radius);
    dict.setValue("max_photon_depth", _max_photon_depth);
    dict.setValue("max_camera_depth", _max_camera_depth);
    return dict;
}

nlohmann::json IntegratorSettings<PhotonMapIntegrator>::toJson() const
{
    static nlohmann::json j;
    j["reduction_factor"] = _reduction_factor;
    j["gather_radius"]    = _gather_radius;
    j["max_photon_depth"] = _max_photon_depth;
    j["max_camera_depth"] = _max_camera_depth;
    j["max_num_photons"]  = _max_num_photons;
    return j;
}

void IntegratorSettings<PhotonMapIntegrator>::fromJson(const nlohmann::json& j)
{
    _reduction_factor = j.value("reduction_factor", 0.7f);
    _gather_radius    = j.value("gather_radius", 0.25f);
    _max_photon_depth = j.value("max_photon_depth", 16);
    _max_camera_depth = j.value("max_camera_depth", 16);
    _max_num_photons  = j.value("max_num_photons", 1000000);
}

}    // namespace atcg