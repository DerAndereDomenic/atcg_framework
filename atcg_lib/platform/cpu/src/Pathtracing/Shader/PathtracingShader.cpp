#include <Pathtracing/PathtracingShader.h>

#include <DataStructure/TorchUtils.h>

#include <Scene/Components.h>
#include <Scene/Entity.h>
#include <Pathtracing/Tracing.h>
#include <Pathtracing/MeshShapeData.cuh>
#include <Pathtracing/Launch.h>

#include <Math/Utils.h>

#include <thread>
#include <mutex>
#include <execution>

namespace atcg
{
PathtracingShader::PathtracingShader() : RaytracingShader()
{
    setTensor("HDR", torch::zeros({1, 1, 3}, atcg::TensorOptions::floatDeviceOptions()));
}

PathtracingShader::~PathtracingShader()
{
    reset();
}

void PathtracingShader::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                           const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    reset();

    if(Renderer::hasSkybox())
    {
        auto skybox_texture = Renderer::getSkyboxTexture();

        _environment_emitter = atcg::make_ref<EnvironmentEmitter>(skybox_texture);
        _environment_emitter->initializePipeline(pipeline, sbt);
    }

    auto view = _scene->getAllEntitiesWith<GeometryComponent, MeshRenderComponent, TransformComponent>();
    for(auto e: view)
    {
        Entity entity(e, _scene.get());

        Material material = entity.getComponent<atcg::MeshRenderComponent>().material;

        auto graph                                    = entity.getComponent<GeometryComponent>().graph;
        atcg::ref_ptr<GASAccelerationStructure> accel = atcg::make_ref<GASAccelerationStructure>(graph);
        accel->initializePipeline(pipeline, sbt);

        atcg::ref_ptr<BSDF> bsdf = nullptr;
        if(material.glass)
        {
            bsdf = atcg::make_ref<RefractiveBSDF>(material);
        }
        else
        {
            bsdf = atcg::make_ref<PBRBSDF>(material);
        }
        bsdf->initializePipeline(pipeline, sbt);
        entity.addOrReplaceComponent<atcg::BSDFComponent>(bsdf);

        entity.addOrReplaceComponent<AccelerationStructureComponent>(accel);

        atcg::ref_ptr<Emitter> emitter = nullptr;
        if(material.emissive)
        {
            glm::mat4 transform = entity.getComponent<atcg::TransformComponent>().getModel();
            auto positions      = graph->getHostPositions();
            auto uvs            = graph->getHostUVs();
            auto faces          = graph->getHostFaces();
            emitter             = atcg::make_ref<MeshEmitter>(accel->getPositions(),
                                                  accel->getUVs(),
                                                  accel->getFaces(),
                                                  transform,
                                                  material);
            emitter->initializePipeline(pipeline, sbt);
        }
        entity.addOrReplaceComponent<atcg::EmitterComponent>(emitter);
        if(emitter) _emitter.push_back(emitter->getVPtrTable());
    }

    if(_environment_emitter)
    {
        _emitter.push_back(_environment_emitter->getVPtrTable());
    }

    auto ptx_file     = "./bin/PathtracingShader.ptx";
    auto raygen_group = pipeline->addRaygenShader({ptx_file, "__raygen__main"});
    auto miss_group   = pipeline->addMissShader({ptx_file, "__miss__main"});
    _raygen_idx       = sbt->addRaygenEntry(raygen_group);
    sbt->addMissEntry(miss_group);
    // TODO: Miss

    for(auto e: view)
    {
        Entity entity(e, _scene.get());

        AccelerationStructureComponent& acc           = entity.getComponent<AccelerationStructureComponent>();
        atcg::ref_ptr<GASAccelerationStructure> accel = std::dynamic_pointer_cast<GASAccelerationStructure>(acc.accel);

        atcg::ref_ptr<BSDF> bsdf       = entity.getComponent<BSDFComponent>().bsdf;
        atcg::ref_ptr<Emitter> emitter = entity.getComponent<EmitterComponent>().emitter;

        MeshShapeData hit_data;
        hit_data.positions    = (glm::vec3*)accel->getPositions().data_ptr();
        hit_data.normals      = (glm::vec3*)accel->getNormals().data_ptr();
        hit_data.uvs          = (glm::vec3*)accel->getUVs().data_ptr();
        hit_data.faces        = (glm::u32vec3*)accel->getFaces().data_ptr();
        hit_data.bsdf         = bsdf->getVPtrTable();
        hit_data.emitter      = emitter ? emitter->getVPtrTable() : nullptr;
        hit_data.num_vertices = accel->getPositions().size(0);
        hit_data.num_faces    = accel->getFaces().size(0);

        sbt->addHitEntry(accel->getHitGroup(), hit_data);
    }

    _accel    = atcg::make_ref<IASAccelerationStructure>(_scene);
    _sbt      = sbt;
    _pipeline = pipeline;

    _sbt->createSBT();
}

void PathtracingShader::reset()
{
    _frame_counter = 0;
}

void PathtracingShader::setCamera(const atcg::ref_ptr<PerspectiveCamera>& camera)
{
    _camera          = camera;
    _inv_camera_view = glm::inverse(camera->getView());
    _fov_y           = camera->getFOV();
}

void PathtracingShader::generateRays(torch::Tensor& output)
{
    auto accumulation_buffer = getTensor("HDR");
    if(accumulation_buffer.ndimension() != 3 || accumulation_buffer.size(0) != output.size(0) ||
       accumulation_buffer.size(1) != output.size(1))
    {
        accumulation_buffer =
            torch::zeros({output.size(0), output.size(1), 3}, atcg::TensorOptions::floatDeviceOptions());
        setTensor("HDR", accumulation_buffer);
    }

    PathtracingParams params;

    glm::mat4 camera_view = _inv_camera_view;
    std::memcpy(params.cam_eye, glm::value_ptr(camera_view[3]), sizeof(glm::vec3));
    std::memcpy(params.U, glm::value_ptr(glm::normalize(camera_view[0])), sizeof(glm::vec3));
    std::memcpy(params.V, glm::value_ptr(glm::normalize(camera_view[1])), sizeof(glm::vec3));
    std::memcpy(params.W, glm::value_ptr(-glm::normalize(camera_view[2])), sizeof(glm::vec3));
    params.image_height        = output.size(0);
    params.image_width         = output.size(1);
    params.accumulation_buffer = (glm::vec3*)accumulation_buffer.data_ptr();
    params.output_image        = (glm::u8vec4*)output.data_ptr();
    params.fov_y               = _fov_y;
    params.emitters            = _emitter.data();
    params.num_emitters        = _emitter.size();
    params.bsdfs               = _bsdfs.data();
    params.num_bsdfs           = _bsdfs.size();
    params.environment_emitter = _environment_emitter ? _environment_emitter->getVPtrTable() : nullptr;
    params.frame_counter       = _frame_counter++;
    params.handle              = _accel.get();

    launch(_pipeline->getPipeline(),
           nullptr,
           &params,
           sizeof(PathtracingParams),
           _sbt->getSBT(_raygen_idx),
           output.size(1),
           output.size(0),
           1);
}
}    // namespace atcg