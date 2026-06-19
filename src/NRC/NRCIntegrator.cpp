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

#ifndef ATCG_HEADLESS
    #include <imgui.h>
#endif

namespace atcg
{
NRCIntegrator::NRCIntegrator(const atcg::ref_ptr<RaytracingContext>& context, const Dictionary& dict)
    : Integrator(context, dict)
{
    initializePipeline(dict);
}

NRCIntegrator::~NRCIntegrator() {}

void NRCIntegrator::initializePipeline(const Dictionary& dict)
{
    _pipeline->addTrianglesHitGroupShader("MeshShape", 0, {"./bin/MeshShape_ptx.ptx", "__closesthit__mesh"}, {});

    auto scene = dict.getValue<atcg::ref_ptr<Scene>>("scene");

    _optix_scene = SceneAdapter(_context, _pipeline, _sbt)
                       .apply(scene, dict.getValue<uint32_t>("width"), dict.getValue<uint32_t>("height"));

    const std::string ptx_raygen_filename = "./bin/NRCIntegrator_ptx.ptx";
    OptixProgramGroup raygen_prog_group   = _pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__rg"});
    OptixProgramGroup miss_prog_group     = _pipeline->addMissShader({ptx_raygen_filename, "__miss__ms"});
    OptixProgramGroup occl_prog_group     = _pipeline->addMissShader({ptx_raygen_filename, "__miss__occlusion"});

    _raygen_index         = _sbt->addRaygenEntry(raygen_prog_group);
    _surface_miss_index   = _sbt->addMissEntry(miss_prog_group);
    _occlusion_miss_index = _sbt->addMissEntry(occl_prog_group);

    _pipeline->createPipeline();
    _sbt->createSBT();
}

void NRCIntegrator::onImGuiRender()
{
#ifndef ATCG_HEADLESS
    ImGui::Begin("NRCIntegrator");
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

    _launch_params.upload(&params);

    _pipeline->launch((CUdeviceptr)_launch_params.get(),
                      sizeof(NRCParams),
                      _sbt->getSBT(_raygen_index),
                      width,
                      height,
                      1,
                      nullptr);

    torch::Tensor output_tensor = _optix_scene->getSensor()->getFilm()->develop();
    in_out_dictionary.setValue("output", output_tensor);
    in_out_dictionary.setValue("entity_ids", output_entities);
}
}    // namespace atcg