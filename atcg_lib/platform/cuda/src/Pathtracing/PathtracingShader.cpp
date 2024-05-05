#include <Pathtracing/PathtracingShader.h>
#include <Renderer/Renderer.h>

#include <Scene/Scene.h>
#include <Scene/Components.h>
#include <Scene/Entity.h>

#include <Pathtracing/Common.h>

#include <Pathtracing/HitGroupData.h>
#include <Pathtracing/EmitterModels.h>
#include <Pathtracing/BSDFModels.h>
#include <Pathtracing/OptixAccelerationStructure.h>

#include <optix_stubs.h>

namespace atcg
{
PathtracingShader::PathtracingShader(OptixDeviceContext context) : OptixRaytracingShader(context)
{
    CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
}

PathtracingShader::~PathtracingShader()
{
    reset();
    CUDA_SAFE_CALL(cudaStreamDestroy(_stream));
}

void PathtracingShader::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
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
    std::vector<TransformComponent> transforms;
    size_t num_instances = 0;
    std::vector<const atcg::EmitterVPtrTable*> emitter_tables;
    for(auto e: view)
    {
        Entity entity(e, _scene.get());

        TransformComponent transform = entity.getComponent<TransformComponent>();

        transforms.push_back(transform);

        Material material = entity.getComponent<atcg::MeshRenderComponent>().material;

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

        entity.addComponent<BSDFComponent>(bsdf);

        atcg::ref_ptr<Graph> graph = entity.getComponent<GeometryComponent>().graph;

        if(!entity.hasComponent<AccelerationStructureComponent>())
        {
            entity.addComponent<AccelerationStructureComponent>();
        }
        AccelerationStructureComponent& acc = entity.getComponent<AccelerationStructureComponent>();

        atcg::ref_ptr<OptixAccelerationStructure> accel = atcg::make_ref<OptixAccelerationStructure>(_context, graph);
        accel->initializePipeline(pipeline, sbt);

        acc.accel = accel;

        atcg::ref_ptr<OptixEmitter> emitter = nullptr;
        if(material.emissive)
        {
            emitter = atcg::make_ref<MeshEmitter>(accel->getPositions(), accel->getFaces(), transform, material);
            emitter->initializePipeline(pipeline, sbt);
        }

        entity.addComponent<EmitterComponent>(emitter);

        if(emitter) emitter_tables.push_back(emitter->getVPtrTable());


        graph->unmapAllPointers();

        ++num_instances;
    }

    if(_environment_emitter)
    {
        emitter_tables.push_back(_environment_emitter->getVPtrTable());
    }

    _emitter_tables = atcg::DeviceBuffer<const atcg::EmitterVPtrTable*>(emitter_tables.size());
    _emitter_tables.upload(emitter_tables.data());

    // IAS
    std::vector<OptixInstance> optix_instances(num_instances);
    int i = 0;
    for(auto e: view)
    {
        Entity entity(e, _scene.get());

        AccelerationStructureComponent& acc             = entity.getComponent<AccelerationStructureComponent>();
        atcg::ref_ptr<OptixAccelerationStructure> accel = acc.accel;

        auto& optix_instance = optix_instances[i];
        memset(&optix_instance, 0, sizeof(OptixInstance));

        optix_instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        optix_instance.instanceId        = i;
        optix_instance.sbtOffset         = i;
        optix_instance.visibilityMask    = 1;
        optix_instance.traversableHandle = accel->getTraversableHandle();

        reinterpret_cast<glm::mat3x4&>(optix_instance.transform) =
            glm::transpose(glm::mat4x3(transforms[i].getModel()));

        ++i;
    }

    atcg::DeviceBuffer<OptixInstance> d_instances(num_instances);
    d_instances.upload(optix_instances.data());

    OptixBuildInput instance_input            = {};
    instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances    = (CUdeviceptr)d_instances.get();
    instance_input.instanceArray.numInstances = static_cast<uint32_t>(num_instances);

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(_context,
                                             &accel_options,
                                             &instance_input,
                                             1,    // num build inputs
                                             &ias_buffer_sizes));

    atcg::DeviceBuffer<uint8_t> d_temp_buffer(ias_buffer_sizes.tempSizeInBytes);
    _ias_buffer = atcg::DeviceBuffer<uint8_t>(ias_buffer_sizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(_context,
                                nullptr,    // CUDA stream
                                &accel_options,
                                &instance_input,
                                1,    // num build inputs
                                (CUdeviceptr)d_temp_buffer.get(),
                                ias_buffer_sizes.tempSizeInBytes,
                                (CUdeviceptr)_ias_buffer.get(),
                                ias_buffer_sizes.outputSizeInBytes,
                                &_ias_handle,
                                nullptr,    // emitted property list
                                0           // num emitted properties
                                ));


    const std::string ptx_raygen_filename = "./build/ptxmodules.dir/Debug/RaygenKernels.ptx";
    OptixProgramGroup raygen_prog_group   = pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__rg"});
    OptixProgramGroup miss_prog_group     = pipeline->addMissShader({ptx_raygen_filename, "__miss__ms"});

    // SBT
    _raygen_index = sbt->addRaygenEntry(raygen_prog_group);
    sbt->addMissEntry(miss_prog_group);

    for(auto e: view)
    {
        Entity entity(e, _scene.get());

        AccelerationStructureComponent& acc             = entity.getComponent<AccelerationStructureComponent>();
        atcg::ref_ptr<OptixAccelerationStructure> accel = acc.accel;

        atcg::ref_ptr<OptixBSDF> bsdf       = entity.getComponent<BSDFComponent>().bsdf;
        atcg::ref_ptr<OptixEmitter> emitter = entity.getComponent<EmitterComponent>().emitter;

        HitGroupData hit_data;
        hit_data.positions = (glm::vec3*)accel->getPositions().data_ptr();
        hit_data.normals   = (glm::vec3*)accel->getNormals().data_ptr();
        hit_data.uvs       = (glm::vec3*)accel->getUVs().data_ptr();
        hit_data.faces     = (glm::u32vec3*)accel->getFaces().data_ptr();
        hit_data.bsdf      = bsdf->getVPtrTable();
        hit_data.emitter   = emitter ? emitter->getVPtrTable() : nullptr;

        sbt->addHitEntry(accel->getHitGroup(), hit_data);
    }
}

void PathtracingShader::reset()
{
    _frame_counter = 0;

    if(!_scene) return;

    auto view = _scene->getAllEntitiesWith<IDComponent>();

    for(auto e: view)
    {
        Entity entity(e, _scene.get());

        if(entity.hasComponent<BSDFComponent>())
        {
            entity.removeComponent<BSDFComponent>();
        }

        if(entity.hasComponent<EmitterComponent>())
        {
            entity.removeComponent<EmitterComponent>();
        }

        if(entity.hasComponent<AccelerationStructureComponent>())
        {
            entity.removeComponent<AccelerationStructureComponent>();
        }
    }
}

void PathtracingShader::setCamera(const atcg::ref_ptr<PerspectiveCamera>& camera)
{
    _inv_camera_view = glm::inverse(camera->getView());
    _fov_y           = camera->getFOV();
}

void PathtracingShader::generateRays(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                     const atcg::ref_ptr<ShaderBindingTable>& sbt,
                                     torch::Tensor& output)
{
    if(_accumulation_buffer.ndimension() != 3 || _accumulation_buffer.size(0) != output.size(0) ||
       _accumulation_buffer.size(1) != output.size(1))
    {
        _accumulation_buffer =
            torch::zeros({output.size(0), output.size(1), 3}, atcg::TensorOptions::floatDeviceOptions());
    }

    Params params;

    memcpy(params.cam_eye, glm::value_ptr(_inv_camera_view[3]), sizeof(glm::vec3));
    memcpy(params.U, glm::value_ptr(glm::normalize(_inv_camera_view[0])), sizeof(glm::vec3));
    memcpy(params.V, glm::value_ptr(glm::normalize(_inv_camera_view[1])), sizeof(glm::vec3));
    memcpy(params.W, glm::value_ptr(-glm::normalize(_inv_camera_view[2])), sizeof(glm::vec3));
    params.fov_y = _fov_y;

    params.accumulation_buffer = (glm::vec3*)_accumulation_buffer.data_ptr();
    params.output_image        = (glm::u8vec4*)output.data_ptr();
    params.image_height        = output.size(0);
    params.image_width         = output.size(1);
    params.handle              = _ias_handle;
    params.frame_counter       = _frame_counter;
    params.num_emitters        = _emitter_tables.size();
    params.emitters            = _emitter_tables.get();

    params.environment_emitter = _environment_emitter ? _environment_emitter->getVPtrTable() : nullptr;

    _launch_params.upload(&params);

    OPTIX_CHECK(optixLaunch(pipeline->getPipeline(),
                            _stream,
                            (CUdeviceptr)_launch_params.get(),
                            sizeof(Params),
                            sbt->getSBT(_raygen_index),
                            output.size(1),
                            output.size(0),
                            1));    // depth

    CUDA_SAFE_CALL(cudaStreamSynchronize(_stream));

    ++_frame_counter;
}


}    // namespace atcg