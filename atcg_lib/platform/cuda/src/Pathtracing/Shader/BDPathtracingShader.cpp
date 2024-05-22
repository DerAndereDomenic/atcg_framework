#include <Pathtracing/Shader/BDPathtracingShader.h>
#include <Renderer/Renderer.h>

#include <Scene/Scene.h>
#include <Scene/Components.h>
#include <Scene/Entity.h>

#include <Pathtracing/Common.h>

#include <Pathtracing/Shape/MeshShape.cuh>
#include <Pathtracing/Emitter/EmitterModels.h>
#include <Pathtracing/BSDF/BSDFModels.h>

#include <optix_stubs.h>

#include <Pathtracing/Payload.h>

namespace atcg
{
BDPathtracingShader::BDPathtracingShader(OptixDeviceContext context) : OptixRaytracingShader(context)
{
    CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));

    setTensor("HDR", torch::zeros({1, 1, 3}, atcg::TensorOptions::floatDeviceOptions()));
}

BDPathtracingShader::~BDPathtracingShader()
{
    reset();
    CUDA_SAFE_CALL(cudaStreamDestroy(_stream));
}

void BDPathtracingShader::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                             const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    reset();
    if(Renderer::hasSkybox())
    {
        auto skybox_texture = Renderer::getSkyboxTexture();

        _environment_emitter = atcg::make_ref<atcg::EnvironmentEmitter>(skybox_texture);
        _environment_emitter->initializePipeline(pipeline, sbt);
    }

    // Extract scene information
    auto view = _scene->getAllEntitiesWith<GeometryComponent, MeshRenderComponent, TransformComponent>();
    std::vector<const atcg::EmitterVPtrTable*> emitter_tables;
    for(auto e: view)
    {
        Entity entity(e, _scene.get());

        Material material = entity.getComponent<atcg::MeshRenderComponent>().material;

        auto graph                                     = entity.getComponent<GeometryComponent>().graph;
        atcg::ref_ptr<MeshAccelerationStructure> accel = atcg::make_ref<MeshAccelerationStructure>(_context, graph);
        accel->initializePipeline(pipeline, sbt);

        atcg::ref_ptr<OptixBSDF> bsdf;
        if(material.glass)
        {
            bsdf = atcg::make_ref<RefractiveBSDF>(material);
            bsdf->initializePipeline(pipeline, sbt);
        }
        else
        {
            bsdf = atcg::make_ref<PBRBSDF>(material);
            bsdf->initializePipeline(pipeline, sbt);
        }

        entity.addOrReplaceComponent<BSDFComponent>(bsdf);

        entity.addOrReplaceComponent<AccelerationStructureComponent>(accel);

        atcg::ref_ptr<OptixEmitter> emitter = nullptr;
        if(material.emissive)
        {
            glm::mat4 transform = entity.getComponent<atcg::TransformComponent>().getModel();
            emitter             = atcg::make_ref<MeshEmitter>(accel->getPositions(),
                                                  accel->getUVs(),
                                                  accel->getFaces(),
                                                  transform,
                                                  material);
            emitter->initializePipeline(pipeline, sbt);
        }

        entity.addOrReplaceComponent<EmitterComponent>(emitter);

        if(emitter) emitter_tables.push_back(emitter->getVPtrTable());
    }

    if(_environment_emitter)
    {
        emitter_tables.push_back(_environment_emitter->getVPtrTable());
    }

    _emitter_tables = atcg::DeviceBuffer<const atcg::EmitterVPtrTable*>(emitter_tables.size());
    _emitter_tables.upload(emitter_tables.data());

    const std::string ptx_raygen_filename = "./bin/PathtracingShader.ptx";
    OptixProgramGroup raygen_prog_group   = pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__rg"});
    OptixProgramGroup miss_prog_group     = pipeline->addMissShader({ptx_raygen_filename, "__miss__ms"});
    OptixProgramGroup occl_prog_group     = pipeline->addMissShader({ptx_raygen_filename, "__miss__occlusion"});

    _raygen_index         = sbt->addRaygenEntry(raygen_prog_group);
    _surface_miss_index   = sbt->addMissEntry(miss_prog_group);
    _occlusion_miss_index = sbt->addMissEntry(occl_prog_group);

    for(auto e: view)
    {
        Entity entity(e, _scene.get());

        AccelerationStructureComponent& acc = entity.getComponent<AccelerationStructureComponent>();
        atcg::ref_ptr<MeshAccelerationStructure> accel =
            std::dynamic_pointer_cast<MeshAccelerationStructure>(acc.accel);

        atcg::ref_ptr<OptixBSDF> bsdf = std::dynamic_pointer_cast<OptixBSDF>(entity.getComponent<BSDFComponent>().bsdf);
        atcg::ref_ptr<OptixEmitter> emitter =
            std::dynamic_pointer_cast<OptixEmitter>(entity.getComponent<EmitterComponent>().emitter);

        MeshShapeData hit_data;
        hit_data.positions = (glm::vec3*)accel->getPositions().data_ptr();
        hit_data.normals   = (glm::vec3*)accel->getNormals().data_ptr();
        hit_data.uvs       = (glm::vec3*)accel->getUVs().data_ptr();
        hit_data.faces     = (glm::u32vec3*)accel->getFaces().data_ptr();
        hit_data.bsdf      = bsdf->getVPtrTable();
        hit_data.emitter   = emitter ? emitter->getVPtrTable() : nullptr;

        sbt->addHitEntry(accel->getHitGroup(), hit_data);
    }

    _accel = atcg::make_ref<IASAccelerationStructure>(_context, _scene);
}

void BDPathtracingShader::reset()
{
    _frame_counter = 0;
}

void BDPathtracingShader::setCamera(const atcg::ref_ptr<PerspectiveCamera>& camera)
{
    _inv_camera_view = glm::inverse(camera->getView());
    _fov_y           = camera->getFOV();
}

void BDPathtracingShader::generateRays(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                       const atcg::ref_ptr<ShaderBindingTable>& sbt,
                                       torch::Tensor& output)
{
    auto accumulation_buffer = getTensor("HDR");
    if(accumulation_buffer.ndimension() != 3 || accumulation_buffer.size(0) != output.size(0) ||
       accumulation_buffer.size(1) != output.size(1))
    {
        accumulation_buffer =
            torch::zeros({output.size(0), output.size(1), 3}, atcg::TensorOptions::floatDeviceOptions());
        setTensor("HDR", accumulation_buffer);
    }

    BDPathtracingParams params;

    memcpy(params.cam_eye, glm::value_ptr(_inv_camera_view[3]), sizeof(glm::vec3));
    memcpy(params.U, glm::value_ptr(glm::normalize(_inv_camera_view[0])), sizeof(glm::vec3));
    memcpy(params.V, glm::value_ptr(glm::normalize(_inv_camera_view[1])), sizeof(glm::vec3));
    memcpy(params.W, glm::value_ptr(-glm::normalize(_inv_camera_view[2])), sizeof(glm::vec3));
    params.fov_y = _fov_y;

    params.accumulation_buffer = (glm::vec3*)accumulation_buffer.data_ptr();
    params.output_image        = (glm::u8vec4*)output.data_ptr();
    params.image_height        = output.size(0);
    params.image_width         = output.size(1);
    params.handle              = _accel->getTraversableHandle();
    params.frame_counter       = _frame_counter;
    params.num_emitters        = _emitter_tables.size();
    params.emitters            = _emitter_tables.get();

    params.environment_emitter = _environment_emitter ? _environment_emitter->getVPtrTable() : nullptr;

    params.surface_trace_params.rayFlags     = OPTIX_RAY_FLAG_NONE;
    params.surface_trace_params.SBToffset    = 0;
    params.surface_trace_params.SBTstride    = 1;
    params.surface_trace_params.missSBTIndex = _surface_miss_index;

    params.occlusion_trace_params.rayFlags  = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
    params.occlusion_trace_params.SBToffset = 0;
    params.occlusion_trace_params.SBTstride = 1;
    params.occlusion_trace_params.missSBTIndex = _occlusion_miss_index;

    _launch_params.upload(&params);

    OPTIX_CHECK(optixLaunch(pipeline->getPipeline(),
                            _stream,
                            (CUdeviceptr)_launch_params.get(),
                            sizeof(BDPathtracingParams),
                            sbt->getSBT(_raygen_index),
                            output.size(1),
                            output.size(0),
                            1));    // depth

    CUDA_SAFE_CALL(cudaStreamSynchronize(_stream));

    ++_frame_counter;
}


}    // namespace atcg