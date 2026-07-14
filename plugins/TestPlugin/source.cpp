#include <stdio.h>
#include <Plugin/Plugin.h>
#include "TestIntegrator.h"

#include <ATCG.h>

#include "TestIntegrator.h"

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
#include "TestMaterial.h"

#ifndef ATCG_HEADLESS
    #include <imgui.h>
#endif

namespace atcg
{
TestIntegrator::TestIntegrator(const atcg::ref_ptr<RaytracingContext>& context, const Dictionary& dict)
    : Integrator(context, dict)
{
    initializePipeline(dict);
}

TestIntegrator::~TestIntegrator() {}

void TestIntegrator::initializePipeline(const Dictionary& dict)
{
    _pipeline->addTrianglesHitGroupShader("MeshShape", 0, {"./bin/MeshShape_ptx.ptx", "__closesthit__mesh"}, {});

    auto scene = dict.getValue<atcg::ref_ptr<Scene>>("scene");

    _optix_scene = SceneAdapter(_context, _pipeline, _sbt)
                       .apply(scene, dict.getValue<uint32_t>("width"), dict.getValue<uint32_t>("height"));

    const std::string ptx_raygen_filename = "./bin/PathtracingIntegrator_ptx.ptx";
    OptixProgramGroup raygen_prog_group   = _pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__rg"});
    OptixProgramGroup miss_prog_group     = _pipeline->addMissShader({ptx_raygen_filename, "__miss__ms"});
    OptixProgramGroup occl_prog_group     = _pipeline->addMissShader({ptx_raygen_filename, "__miss__occlusion"});

    _raygen_index         = _sbt->addRaygenEntry(raygen_prog_group);
    _surface_miss_index   = _sbt->addMissEntry(miss_prog_group);
    _occlusion_miss_index = _sbt->addMissEntry(occl_prog_group);

    _pipeline->createPipeline();
    _sbt->createSBT();
}

void TestIntegrator::onImGuiRender()
{
#ifndef ATCG_HEADLESS
    ImGui::Begin("TestIntegrator");
    for(auto shape: _optix_scene->getShapes())
    {
        shape->onImGuiRender();
    }
    ImGui::End();
#endif
}

void TestIntegrator::reset()
{
    _frame_counter = 0;
    _optix_scene->getSensor()->getFilm()->clear();
    _optix_scene->getSensor()->markDirty();
}

void TestIntegrator::generateRays(Dictionary& in_out_dictionary)
{
    uint32_t width  = _optix_scene->getSensor()->getFilm()->getWidth();
    uint32_t height = _optix_scene->getSensor()->getFilm()->getHeight();

    torch::Tensor output_entities = torch::zeros({height, width}, atcg::TensorOptions::int32DeviceOptions());

    PathtracingParams params;

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
                      sizeof(PathtracingParams),
                      _sbt->getSBT(_raygen_index),
                      width,
                      height,
                      1);

    torch::Tensor output_tensor = _optix_scene->getSensor()->getFilm()->develop();
    in_out_dictionary.setValue("output", output_tensor);
    in_out_dictionary.setValue("entity_ids", output_entities);
}

DiffuseMaterial::DiffuseMaterial() : atcg::Material("Diffuse")
{
    atcg::TextureSpecification spec_diffuse;
    spec_diffuse.width  = 1;
    spec_diffuse.height = 1;
    glm::u8vec4 white(255);
    _diffuse_texture = atcg::Texture2D::create(&white, spec_diffuse);
}

void DiffuseMaterial::uploadMaterial(atcg::RendererSystem* renderer, const atcg::ref_ptr<atcg::Shader>& shader)
{
    ATCG_ASSERT(!_uploaded, "Material was already uploaded");

    uint32_t diffuse_id = renderer->popTextureID();
    atcg::GraphicsCommand::bindTexture(diffuse_id, getDiffuseTexture());
    shader->setInt("texture_diffuse", diffuse_id);
    _used_texture_ids[0] = diffuse_id;

    uint32_t normal_id   = renderer->popTextureID();
    _used_texture_ids[1] = normal_id;

    uint32_t roughness_id = renderer->popTextureID();
    _used_texture_ids[2]  = roughness_id;

    uint32_t metallic_id = renderer->popTextureID();
    _used_texture_ids[3] = metallic_id;

    uint32_t ior_id      = renderer->popTextureID();
    _used_texture_ids[4] = ior_id;

    // Select shading functions
    shader->selectSubroutine("sr_eval_brdf", "eval_brdf_diffuse");
    shader->selectSubroutine("sr_image_based_lighting", "image_based_lighting_diffuse");

    _uploaded = true;
}

atcg::ref_ptr<atcg::Material> DiffuseMaterial::clone() const
{
    atcg::ref_ptr<DiffuseMaterial> material = atcg::make_ref<DiffuseMaterial>();

    material->setDiffuseTexture(std::dynamic_pointer_cast<atcg::Texture2D>(getDiffuseTexture()->clone()));

    return material;
}

void DiffuseMaterial::setDiffuseColor(const glm::vec3& color)
{
    atcg::TextureSpecification spec_diffuse;
    spec_diffuse.width  = 1;
    spec_diffuse.height = 1;
    glm::u8vec4 color_quant((uint8_t)(color[0] * 255.0f),
                            (uint8_t)(color[1] * 255.0f),
                            (uint8_t)(color[2] * 255.0f),
                            (uint8_t)(255.0f));
    _diffuse_texture = atcg::Texture2D::create(&color_quant, spec_diffuse);
}

bool MaterialGUIRenderer<DiffuseMaterial>::renderGUI(const atcg::ref_ptr<DiffuseMaterial>& material,
                                                     const std::string& key,
                                                     bool& deactivated)
{
    bool updated = false;

    auto diffuse = material->getDiffuseTexture()->getData(atcg::CPU);

    float color[4] = {diffuse.index({0, 0, 0}).item<float>() / 255.0f,
                      diffuse.index({0, 0, 1}).item<float>() / 255.0f,
                      diffuse.index({0, 0, 2}).item<float>() / 255.0f,
                      diffuse.index({0, 0, 3}).item<float>() / 255.0f};

    if(ImGui::ColorEdit4(("Diffuse##" + key).c_str(), color))
    {
        glm::vec4 new_color = glm::make_vec4(color);
        material->setDiffuseColor(new_color);
        updated = true;
    }

    deactivated = ImGui::IsItemDeactivated() || deactivated;
    return updated;
}

}    // namespace atcg

ATCG_PLUGIN_LIBRARY();

extern "C" __declspec(dllexport) void registerPlugin(atcg::PluginRegistry& registry)
{
    registry.registerMaterial<atcg::DiffuseMaterial>("Diffuse");
    registry.registerIntegrator<atcg::TestIntegrator>("TestIntegrator");
}

extern "C" __declspec(dllexport) void registerPythonBindings(pybind11::module& m)
{
    ATCG_DEBUG("Registering Python bindings for TestPlugin");
    auto test_plugin = m.def_submodule("TestPlugin");
    test_plugin.def("Test", []() { printf("TestPlugin::Test() called\n"); });
}