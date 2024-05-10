#include <Pathtracing/OptixAccelerationStructure.h>
#include <Pathtracing/Common.h>

#include <Scene/Components.h>
#include <Scene/Entity.h>

#include <Pathtracing/HitGroupData.h>

#include <optix_stubs.h>

namespace atcg
{
MeshAccelerationStructure::MeshAccelerationStructure(OptixDeviceContext context, const atcg::ref_ptr<Graph>& graph)
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
    _ast_buffer = atcg::DeviceBuffer<uint8_t>(gas_buffer_sizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(context,
                                0,    // CUDA stream
                                &accel_options,
                                &triangle_input,
                                1,    // num build inputs
                                (CUdeviceptr)d_temp_buffer_gas.get(),
                                gas_buffer_sizes.tempSizeInBytes,
                                (CUdeviceptr)_ast_buffer.get(),
                                gas_buffer_sizes.outputSizeInBytes,
                                &_handle,    // Output handle to the struct
                                nullptr,     // emitted property list
                                0));         // num emitted properties

    graph->unmapAllPointers();
}

MeshAccelerationStructure::~MeshAccelerationStructure() {}

void MeshAccelerationStructure::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
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

    // IAS
    std::vector<OptixInstance> optix_instances;
    int num_instances = 0;
    for(auto e: view)
    {
        Entity entity(e, _scene.get());

        AccelerationStructureComponent& acc = entity.getComponent<AccelerationStructureComponent>();
        atcg::ref_ptr<OptixAccelerationStructure> accel =
            std::dynamic_pointer_cast<OptixAccelerationStructure>(acc.accel);

        OptixInstance optix_instance = {0};
        memset(&optix_instance, 0, sizeof(OptixInstance));

        optix_instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        optix_instance.instanceId        = num_instances;
        optix_instance.sbtOffset         = num_instances;
        optix_instance.visibilityMask    = 1;
        optix_instance.traversableHandle = accel->getTraversableHandle();

        glm::mat4 transform = entity.getComponent<atcg::TransformComponent>().getModel();

        reinterpret_cast<glm::mat3x4&>(optix_instance.transform) = glm::transpose(glm::mat4x3(transform));

        ++num_instances;

        optix_instances.push_back(optix_instance);
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
    _ast_buffer = atcg::DeviceBuffer<uint8_t>(ias_buffer_sizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(context,
                                nullptr,    // CUDA stream
                                &accel_options,
                                &instance_input,
                                1,    // num build inputs
                                (CUdeviceptr)d_temp_buffer.get(),
                                ias_buffer_sizes.tempSizeInBytes,
                                (CUdeviceptr)_ast_buffer.get(),
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
}
}    // namespace atcg