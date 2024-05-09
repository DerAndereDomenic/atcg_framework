#include <Pathtracing/OptixAccelerationStructure.h>
#include <Pathtracing/Common.h>

#include <Scene/Components.h>
#include <Scene/Entity.h>

#include <Pathtracing/HitGroupData.h>

#include <optix_stubs.h>

namespace atcg
{
GASAccelerationStructure::GASAccelerationStructure(OptixDeviceContext context, const atcg::ref_ptr<Graph>& graph)
{
    _positions = graph->getDevicePositions().clone();
    _normals   = graph->getDeviceNormals().clone();
    _uvs       = graph->getDeviceUVs().clone();
    _faces     = graph->getDeviceFaces().clone();

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    const uint32_t triangle_input_flags[1]     = {OPTIX_GEOMETRY_FLAG_NONE};
    OptixBuildInput triangle_input             = {};
    triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices   = _positions.size(0);
    CUdeviceptr ptr                            = (CUdeviceptr)_positions.data_ptr();
    triangle_input.triangleArray.vertexBuffers = &ptr;
    triangle_input.triangleArray.flags         = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    triangle_input.triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.numIndexTriplets = _faces.size(0);
    triangle_input.triangleArray.indexBuffer      = (CUdeviceptr)_faces.data_ptr();

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &triangle_input, 1, &gas_buffer_sizes));

    atcg::DeviceBuffer<uint8_t> d_temp_buffer_gas(gas_buffer_sizes.tempSizeInBytes);
    _gas_buffer = atcg::DeviceBuffer<uint8_t>(gas_buffer_sizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(context,
                                0,    // CUDA stream
                                &accel_options,
                                &triangle_input,
                                1,    // num build inputs
                                (CUdeviceptr)d_temp_buffer_gas.get(),
                                gas_buffer_sizes.tempSizeInBytes,
                                (CUdeviceptr)_gas_buffer.get(),
                                gas_buffer_sizes.outputSizeInBytes,
                                &_handle,    // Output handle to the struct
                                nullptr,     // emitted property list
                                0));         // num emitted properties
}

GASAccelerationStructure::~GASAccelerationStructure() {}

void GASAccelerationStructure::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                                  const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_raygen_filename = "./bin/MeshKernels.ptx";
    _hit_group = pipeline->addTrianglesHitGroupShader({ptx_raygen_filename, "__closesthit__mesh"}, {});
}

IASAccelerationStructure::IASAccelerationStructure(OptixDeviceContext context, const atcg::ref_ptr<Scene>& scene)
    : _scene(scene)
{
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

        atcg::ref_ptr<Graph> graph = entity.getComponent<GeometryComponent>().graph;

        if(!entity.hasComponent<AccelerationStructureComponent>())
        {
            entity.addOrReplaceComponent<AccelerationStructureComponent>();
        }
        AccelerationStructureComponent& acc = entity.getComponent<AccelerationStructureComponent>();

        atcg::ref_ptr<GASAccelerationStructure> accel = atcg::make_ref<GASAccelerationStructure>(context, graph);

        acc.accel = accel;

        graph->unmapAllPointers();

        ++num_instances;
    }

    // IAS
    std::vector<OptixInstance> optix_instances(num_instances);
    int i = 0;
    for(auto e: view)
    {
        Entity entity(e, _scene.get());

        AccelerationStructureComponent& acc           = entity.getComponent<AccelerationStructureComponent>();
        atcg::ref_ptr<GASAccelerationStructure> accel = std::dynamic_pointer_cast<GASAccelerationStructure>(acc.accel);

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
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context,
                                             &accel_options,
                                             &instance_input,
                                             1,    // num build inputs
                                             &ias_buffer_sizes));

    atcg::DeviceBuffer<uint8_t> d_temp_buffer(ias_buffer_sizes.tempSizeInBytes);
    _ias_buffer = atcg::DeviceBuffer<uint8_t>(ias_buffer_sizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(context,
                                nullptr,    // CUDA stream
                                &accel_options,
                                &instance_input,
                                1,    // num build inputs
                                (CUdeviceptr)d_temp_buffer.get(),
                                ias_buffer_sizes.tempSizeInBytes,
                                (CUdeviceptr)_ias_buffer.get(),
                                ias_buffer_sizes.outputSizeInBytes,
                                &_handle,
                                nullptr,    // emitted property list
                                0           // num emitted properties
                                ));
}

IASAccelerationStructure::~IASAccelerationStructure() {}

void IASAccelerationStructure::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                                  const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    auto view = _scene->getAllEntitiesWith<GeometryComponent, MeshRenderComponent, TransformComponent>();
    for(auto e: view)
    {
        Entity entity(e, _scene.get());

        AccelerationStructureComponent& acc           = entity.getComponent<AccelerationStructureComponent>();
        atcg::ref_ptr<GASAccelerationStructure> accel = std::dynamic_pointer_cast<GASAccelerationStructure>(acc.accel);
        accel->initializePipeline(pipeline, sbt);

        atcg::ref_ptr<OptixBSDF> bsdf = std::dynamic_pointer_cast<OptixBSDF>(entity.getComponent<BSDFComponent>().bsdf);
        atcg::ref_ptr<OptixEmitter> emitter =
            std::dynamic_pointer_cast<OptixEmitter>(entity.getComponent<EmitterComponent>().emitter);

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
}    // namespace atcg