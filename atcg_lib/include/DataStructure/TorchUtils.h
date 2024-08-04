#pragma once

#include <DataStructure/GraphDefinitions.h>
#include <torch/types.h>
#include <Renderer/Buffer.h>

namespace atcg
{

/// These types are used to easily switch between compilation with and without CUDA

ATCG_CONSTEXPR ATCG_INLINE torch::DeviceType CPU = torch::kCPU;
#ifdef ATCG_CUDA_BACKEND
ATCG_CONSTEXPR ATCG_INLINE torch::DeviceType GPU = torch::kCUDA;
#else
ATCG_CONSTEXPR ATCG_INLINE torch::DeviceType GPU = torch::kCPU;
#endif

/**
 * @brief A namespace for quick access to commonly used tensor options
 */
namespace TensorOptions
{
ATCG_INLINE torch::TensorOptions uint8HostOptions()
{
    return torch::TensorOptions {}.dtype(torch::kUInt8).device(atcg::CPU);
}

ATCG_INLINE torch::TensorOptions int8HostOptions()
{
    return torch::TensorOptions {}.dtype(torch::kInt8).device(atcg::CPU);
}

ATCG_INLINE torch::TensorOptions int16HostOptions()
{
    return torch::TensorOptions {}.dtype(torch::kInt16).device(atcg::CPU);
}

ATCG_INLINE torch::TensorOptions int32HostOptions()
{
    return torch::TensorOptions {}.dtype(torch::kInt32).device(atcg::CPU);
}

ATCG_INLINE torch::TensorOptions int64HostOptions()
{
    return torch::TensorOptions {}.dtype(torch::kInt64).device(atcg::CPU);
}

ATCG_INLINE torch::TensorOptions floatHostOptions()
{
    return torch::TensorOptions {}.dtype(torch::kFloat32).device(atcg::CPU);
}

ATCG_INLINE torch::TensorOptions doubleHostOptions()
{
    return torch::TensorOptions {}.dtype(torch::kFloat64).device(atcg::CPU);
}

ATCG_INLINE torch::TensorOptions int8DeviceOptions()
{
    return torch::TensorOptions {}.dtype(torch::kInt8).device(atcg::GPU);
}

ATCG_INLINE torch::TensorOptions uint8DeviceOptions()
{
    return torch::TensorOptions {}.dtype(torch::kUInt8).device(atcg::GPU);
}

ATCG_INLINE torch::TensorOptions int16DeviceOptions()
{
    return torch::TensorOptions {}.dtype(torch::kInt16).device(atcg::GPU);
}

ATCG_INLINE torch::TensorOptions int32DeviceOptions()
{
    return torch::TensorOptions {}.dtype(torch::kInt32).device(atcg::GPU);
}

ATCG_INLINE torch::TensorOptions int64DeviceOptions()
{
    return torch::TensorOptions {}.dtype(torch::kInt64).device(atcg::GPU);
}

ATCG_INLINE torch::TensorOptions floatDeviceOptions()
{
    return torch::TensorOptions {}.dtype(torch::kFloat32).device(atcg::GPU);
}

ATCG_INLINE torch::TensorOptions doubleDeviceOptions()
{
    return torch::TensorOptions {}.dtype(torch::kFloat64).device(atcg::GPU);
}

template<typename T>
ATCG_INLINE torch::TensorOptions HostOptions();

template<typename T>
ATCG_INLINE torch::TensorOptions DeviceOptions();

template<>
ATCG_INLINE torch::TensorOptions HostOptions<int8_t>()
{
    return int8HostOptions();
}

template<>
ATCG_INLINE torch::TensorOptions HostOptions<uint8_t>()
{
    return uint8HostOptions();
}


template<>
ATCG_INLINE torch::TensorOptions HostOptions<int16_t>()
{
    return int16HostOptions();
}

template<>
ATCG_INLINE torch::TensorOptions HostOptions<int32_t>()
{
    return int32HostOptions();
}

template<>
ATCG_INLINE torch::TensorOptions HostOptions<int64_t>()
{
    return int64HostOptions();
}

template<>
ATCG_INLINE torch::TensorOptions HostOptions<float>()
{
    return floatHostOptions();
}

template<>
ATCG_INLINE torch::TensorOptions HostOptions<double>()
{
    return doubleHostOptions();
}

template<>
ATCG_INLINE torch::TensorOptions DeviceOptions<int8_t>()
{
    return int8DeviceOptions();
}

template<>
ATCG_INLINE torch::TensorOptions DeviceOptions<uint8_t>()
{
    return uint8DeviceOptions();
}

template<>
ATCG_INLINE torch::TensorOptions DeviceOptions<int16_t>()
{
    return int16DeviceOptions();
}

template<>
ATCG_INLINE torch::TensorOptions DeviceOptions<int32_t>()
{
    return int32DeviceOptions();
}

template<>
ATCG_INLINE torch::TensorOptions DeviceOptions<int64_t>()
{
    return int64DeviceOptions();
}

template<>
ATCG_INLINE torch::TensorOptions DeviceOptions<float>()
{
    return floatDeviceOptions();
}

template<>
ATCG_INLINE torch::TensorOptions DeviceOptions<double>()
{
    return doubleDeviceOptions();
}

}    // namespace TensorOptions

/**
 * @brief Create a tensor from a pointer.
 * The tensor does not take ownership of the memory. The pointer has to be on the same device as specified by the passed
 * options
 *
 * @param pointer The data pointer
 * @param size The size
 * @param options The options
 *
 * @return A tensor that wraps around the data pointer
 */
ATCG_INLINE torch::Tensor
createTensorFromPointer(void* pointer, const at::IntArrayRef& size, const torch::TensorOptions& options)
{
    return torch::from_blob(pointer, size, options);
}

/**
 * @brief Create a tensor from a pointer.
 * The tensor does not take ownership of the memory. The pointer has to be on the same device as specified by the passed
 * options
 *
 * @param pointer The data pointer
 * @param size The size
 * @param stride The stride
 * @param options The options
 *
 * @return A tensor that wraps around the data pointer
 */
ATCG_INLINE torch::Tensor createTensorFromPointer(void* pointer,
                                                  const at::IntArrayRef& size,
                                                  const at::IntArrayRef& stride,
                                                  const torch::TensorOptions& options)
{
    return torch::from_blob(pointer, size, stride, options);
}

/**
 * @brief Create a tensor from a pointer.
 * The tensor does not take ownership of the memory. The pointer has to describe a host memory address.
 *
 * @tparam T The datatype
 * @param pointer The data pointer
 * @param size The size
 *
 * @return A tensor that wraps around the data pointer
 */
template<typename T>
ATCG_INLINE torch::Tensor createHostTensorFromPointer(T* pointer, const at::IntArrayRef& size)
{
    return createTensorFromPointer(pointer, size, TensorOptions::HostOptions<T>());
}

/**
 * @brief Create a tensor from a pointer.
 * The tensor does not take ownership of the memory. The pointer has to describe a host memory address.
 *
 * @tparam T The datatype
 * @param pointer The data pointer
 * @param size The size
 * @param stride The stride
 *
 * @return A tensor that wraps around the data pointer
 */
template<typename T>
ATCG_INLINE torch::Tensor
createHostTensorFromPointer(T* pointer, const at::IntArrayRef& size, const at::IntArrayRef& stride)
{
    return createTensorFromPointer(pointer, size, stride, TensorOptions::HostOptions<T>());
}

/**
 * @brief Create a tensor from a pointer.
 * The tensor does not take ownership of the memory. The pointer has to describe a device memory address.
 *
 * @tparam T The datatype
 * @param pointer The data pointer
 * @param size The size
 *
 * @return A tensor that wraps around the data pointer
 */
template<typename T>
ATCG_INLINE torch::Tensor createDeviceTensorFromPointer(T* pointer, const at::IntArrayRef& size)
{
    return createTensorFromPointer(pointer, size, TensorOptions::DeviceOptions<T>());
}

/**
 * @brief Create a tensor from a pointer.
 * The tensor does not take ownership of the memory. The pointer has to describe a device memory address.
 *
 * @tparam T The datatype
 * @param pointer The data pointer
 * @param size The size
 * @param stride The stride
 *
 * @return A tensor that wraps around the data pointer
 */
template<typename T>
ATCG_INLINE torch::Tensor
createDeviceTensorFromPointer(T* pointer, const at::IntArrayRef& size, const at::IntArrayRef& stride)
{
    return createTensorFromPointer(pointer, size, stride, TensorOptions::DeviceOptions<T>());
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership.
 * This function maps the host pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, atcg::VertexSpecification::VERTEX_SIZE) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
ATCG_INLINE torch::Tensor getVertexBufferAsHostTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getHostPointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createHostTensorFromPointer(vertex_buffer, {num_vertices, atcg::VertexSpecification::VERTEX_SIZE});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership.
 * This function maps the device pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, atcg::VertexSpecification::VERTEX_SIZE) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
ATCG_INLINE torch::Tensor getVertexBufferAsDeviceTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getDevicePointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createDeviceTensorFromPointer(vertex_buffer, {num_vertices, atcg::VertexSpecification::VERTEX_SIZE});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex positions.
 * This function maps the host pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, atcg::VertexSpecification::POSITION_END -
 * atcg::VertexSpecification::POSITION_BEGIN) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
ATCG_INLINE torch::Tensor getPositionsAsHostTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getHostPointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createHostTensorFromPointer(
        vertex_buffer,
        {num_vertices, atcg::VertexSpecification::POSITION_END - atcg::VertexSpecification::POSITION_BEGIN},
        {atcg::VertexSpecification::VERTEX_SIZE, 1});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex positions.
 * This function maps the device pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, atcg::VertexSpecification::POSITION_END -
 * atcg::VertexSpecification::POSITION_BEGIN) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
ATCG_INLINE torch::Tensor getPositionsAsDeviceTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getDevicePointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createDeviceTensorFromPointer(
        vertex_buffer,
        {num_vertices, atcg::VertexSpecification::POSITION_END - atcg::VertexSpecification::POSITION_BEGIN},
        {atcg::VertexSpecification::VERTEX_SIZE, 1});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex colors.
 * This function maps the host pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, atcg::VertexSpecification::COLOR_END -
 * atcg::VertexSpecification::COLOR_BEGIN) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
ATCG_INLINE torch::Tensor getColorsAsHostTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getHostPointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createHostTensorFromPointer(
        vertex_buffer + atcg::VertexSpecification::COLOR_BEGIN,
        {num_vertices, atcg::VertexSpecification::COLOR_END - atcg::VertexSpecification::COLOR_BEGIN},
        {atcg::VertexSpecification::VERTEX_SIZE, 1});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex colors.
 * This function maps the device pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, atcg::VertexSpecification::COLOR_END -
 * atcg::VertexSpecification::COLOR_BEGIN) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
ATCG_INLINE torch::Tensor getColorsAsDeviceTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getDevicePointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createDeviceTensorFromPointer(
        vertex_buffer + atcg::VertexSpecification::COLOR_BEGIN,
        {num_vertices, atcg::VertexSpecification::COLOR_END - atcg::VertexSpecification::COLOR_BEGIN},
        {atcg::VertexSpecification::VERTEX_SIZE, 1});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex normals.
 * This function maps the host pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, atcg::VertexSpecification::NORMAL_END -
 * atcg::VertexSpecification::NORMAL_BEGIN) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
ATCG_INLINE torch::Tensor getNormalsAsHostTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getHostPointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createHostTensorFromPointer(
        vertex_buffer + atcg::VertexSpecification::NORMAL_BEGIN,
        {num_vertices, atcg::VertexSpecification::NORMAL_END - atcg::VertexSpecification::NORMAL_BEGIN},
        {atcg::VertexSpecification::VERTEX_SIZE, 1});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex normals.
 * This function maps the device pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, atcg::VertexSpecification::NORMAL_END -
 * atcg::VertexSpecification::NORMAL_BEGIN) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
ATCG_INLINE torch::Tensor getNormalsAsDeviceTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getDevicePointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createDeviceTensorFromPointer(
        vertex_buffer + atcg::VertexSpecification::NORMAL_BEGIN,
        {num_vertices, atcg::VertexSpecification::NORMAL_END - atcg::VertexSpecification::NORMAL_BEGIN},
        {atcg::VertexSpecification::VERTEX_SIZE, 1});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex tangents.
 * This function maps the host pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, atcg::VertexSpecification::TANGENT_END -
 * atcg::VertexSpecification::TANGENT_BEGIN) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
ATCG_INLINE torch::Tensor getTangentsAsHostTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getHostPointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createHostTensorFromPointer(
        vertex_buffer + atcg::VertexSpecification::TANGENT_BEGIN,
        {num_vertices, atcg::VertexSpecification::TANGENT_END - atcg::VertexSpecification::TANGENT_BEGIN},
        {atcg::VertexSpecification::VERTEX_SIZE, 1});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex tangents.
 * This function maps the device pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, atcg::VertexSpecification::TANGENT_END -
 * atcg::VertexSpecification::TANGENT_BEGIN) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
ATCG_INLINE torch::Tensor getTangentsAsDeviceTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getDevicePointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createDeviceTensorFromPointer(
        vertex_buffer + atcg::VertexSpecification::TANGENT_BEGIN,
        {num_vertices, atcg::VertexSpecification::TANGENT_END - atcg::VertexSpecification::TANGENT_BEGIN},
        {atcg::VertexSpecification::VERTEX_SIZE, 1});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex uvs.
 * This function maps the host pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, atcg::VertexSpecification::UV_END -
 * atcg::VertexSpecification::UV_BEGIN) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
ATCG_INLINE torch::Tensor getUVsAsHostTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getHostPointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createHostTensorFromPointer(
        vertex_buffer + atcg::VertexSpecification::UV_BEGIN,
        {num_vertices, atcg::VertexSpecification::UV_END - atcg::VertexSpecification::UV_BEGIN},
        {atcg::VertexSpecification::VERTEX_SIZE, 1});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex uvs.
 * This function maps the device pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, atcg::VertexSpecification::UV_END -
 * atcg::VertexSpecification::UV_BEGIN) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
ATCG_INLINE torch::Tensor getUVsAsDeviceTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getDevicePointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createDeviceTensorFromPointer(
        vertex_buffer + atcg::VertexSpecification::UV_BEGIN,
        {num_vertices, atcg::VertexSpecification::UV_END - atcg::VertexSpecification::UV_BEGIN},
        {atcg::VertexSpecification::VERTEX_SIZE, 1});
}

}    // namespace atcg