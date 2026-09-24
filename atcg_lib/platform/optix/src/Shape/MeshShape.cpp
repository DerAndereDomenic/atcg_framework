#include <Shape/MeshShape.h>

#include <Core/Common.h>
#include <Shape/MeshShapeSampler.h>
#include <Shape/MeshKernels.h>

#include <optix_stubs.h>

namespace atcg
{

namespace detail
{
std::tuple<torch::Tensor, torch::Tensor> makeIndexedAttribute(const torch::Tensor& attribute,
                                                              const torch::Tensor& indices)
{
    auto result = torch::unique_dim(attribute, 0, true, true, false);

    torch::Tensor unique_values = std::get<0>(result);
    torch::Tensor inverse       = std::get<1>(result);

    auto faces = indices.to(torch::kInt64);

    torch::Tensor new_faces = inverse.index_select(0, faces.reshape({-1})).reshape_as(faces).to(torch::kUInt32);
    return std::make_tuple(unique_values, new_faces);
}
}    // namespace detail

MeshShape::MeshShape(const Dictionary& dict)
{
    atcg::ref_ptr<Graph> mesh = dict.getValue<atcg::ref_ptr<Graph>>("mesh");

    auto positions = mesh->getDevicePositions().clone();
    auto normals   = mesh->getDeviceNormals().clone();
    auto colors    = mesh->getDeviceColors().clone();
    auto uvs       = mesh->getDeviceUVs().clone();
    auto faces     = mesh->getDeviceFaces().clone();

    std::tie(_positions, _faces_3d)    = detail::makeIndexedAttribute(positions, faces);
    std::tie(_normals, _faces_normals) = detail::makeIndexedAttribute(normals, faces);
    std::tie(_colors, _faces_color)    = detail::makeIndexedAttribute(colors, faces);
    std::tie(_uvs, _faces_uv)          = detail::makeIndexedAttribute(uvs, faces);

    std::tie(_edges, _edge_faces) = computeMeshEdges(_faces_3d);

    mesh->unmapAllPointers();

    MeshShapeData data;
    data.positions     = (glm::vec3*)_positions.data_ptr();
    data.normals       = (glm::vec3*)_normals.data_ptr();
    data.colors        = (glm::vec3*)_colors.data_ptr();
    data.uvs           = (glm::vec3*)_uvs.data_ptr();
    data.faces_3d      = (glm::u32vec3*)_faces_3d.data_ptr();
    data.faces_uv      = (glm::u32vec3*)_faces_uv.data_ptr();
    data.faces_normals = (glm::u32vec3*)_faces_normals.data_ptr();
    data.faces_color   = (glm::u32vec3*)_faces_color.data_ptr();

    _data.upload(&data);
    _shape_data = _data.get();
}

MeshShape::~MeshShape() {}

void MeshShape::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                   const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    // Nothign to do
}

void MeshShape::prepareAccelerationStructure(const atcg::ref_ptr<RaytracingContext>& context)
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
    triangle_input.triangleArray.numIndexTriplets = _faces_3d.size(0);
    triangle_input.triangleArray.indexBuffer      = (CUdeviceptr)_faces_3d.data_ptr();

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context->getContextHandle(),
                                             &accel_options,
                                             &triangle_input,
                                             1,
                                             &gas_buffer_sizes));

    atcg::DeviceBuffer<uint8_t> d_temp_buffer_gas(gas_buffer_sizes.tempSizeInBytes);
    _ast_buffer = atcg::DeviceBuffer<uint8_t>(gas_buffer_sizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(context->getContextHandle(),
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

atcg::ref_ptr<ShapeSampler> MeshShape::createSampler(const glm::mat4& transform)
{
    atcg::Dictionary dict;
    dict.setValue("transform", transform);
    atcg::ref_ptr<Shape> shape = shared_from_this();
    dict.setValue("shape", shape);
    return atcg::make_ref<MeshShapeSampler>(dict);
}
}    // namespace atcg