#include <Shape/MeshShape.h>

#include <Core/Common.h>

namespace atcg
{
MeshShape::MeshShape(const atcg::ref_ptr<Graph>& mesh)
{
    _positions = mesh->getDevicePositions().clone();
    _normals   = mesh->getDeviceNormals().clone();
    _uvs       = mesh->getDeviceUVs().clone();
    _faces     = mesh->getDeviceFaces().clone();
    mesh->unmapAllPointers();

    MeshShapeData data;
    data.positions = (glm::vec3*)_positions.data_ptr();
    data.normals   = (glm::vec3*)_normals.data_ptr();
    data.uvs       = (glm::vec3*)_uvs.data_ptr();
    data.faces     = (glm::u32vec3*)_faces.data_ptr();

    _data.upload(&data);
    _shape_data = _data.get();
}

MeshShape::~MeshShape() {}

void MeshShape::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                   const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_raygen_filename = "./bin/MeshShape_ptx.ptx";
    _hit_group = pipeline->addTrianglesHitGroupShader({ptx_raygen_filename, "__closesthit__mesh"}, {});
}

void MeshShape::prepareAccelerationStructure(OptixDeviceContext context)
{
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
                                &_ast_handle,    // Output handle to the struct
                                nullptr,         // emitted property list
                                0));             // num emitted properties
}
}    // namespace atcg