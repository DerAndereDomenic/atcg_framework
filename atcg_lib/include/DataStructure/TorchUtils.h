#pragma once

#include <DataStructure/Graph.h>
#include <torch/types.h>

namespace atcg
{

/**
 * @brief A namespace for quick access to commonly used tensor options
 */
namespace TensorOptions
{
inline torch::TensorOptions uint8HostOptions()
{
    return torch::TensorOptions {}.dtype(torch::kUInt8).device(torch::kCPU);
}

inline torch::TensorOptions int8HostOptions()
{
    return torch::TensorOptions {}.dtype(torch::kInt8).device(torch::kCPU);
}

inline torch::TensorOptions int16HostOptions()
{
    return torch::TensorOptions {}.dtype(torch::kInt16).device(torch::kCPU);
}

inline torch::TensorOptions int32HostOptions()
{
    return torch::TensorOptions {}.dtype(torch::kInt32).device(torch::kCPU);
}

inline torch::TensorOptions int64HostOptions()
{
    return torch::TensorOptions {}.dtype(torch::kInt64).device(torch::kCPU);
}

inline torch::TensorOptions floatHostOptions()
{
    return torch::TensorOptions {}.dtype(torch::kFloat32).device(torch::kCPU);
}

inline torch::TensorOptions doubleHostOptions()
{
    return torch::TensorOptions {}.dtype(torch::kFloat64).device(torch::kCPU);
}

inline torch::TensorOptions int8DeviceOptions()
{
    return torch::TensorOptions {}.dtype(torch::kInt8).device(torch::kCUDA);
}

inline torch::TensorOptions uint8DeviceOptions()
{
    return torch::TensorOptions {}.dtype(torch::kUInt8).device(torch::kCUDA);
}

inline torch::TensorOptions int16DeviceOptions()
{
    return torch::TensorOptions {}.dtype(torch::kInt16).device(torch::kCUDA);
}

inline torch::TensorOptions int32DeviceOptions()
{
    return torch::TensorOptions {}.dtype(torch::kInt32).device(torch::kCUDA);
}

inline torch::TensorOptions int64DeviceOptions()
{
    return torch::TensorOptions {}.dtype(torch::kInt64).device(torch::kCUDA);
}

inline torch::TensorOptions floatDeviceOptions()
{
    return torch::TensorOptions {}.dtype(torch::kFloat32).device(torch::kCUDA);
}

inline torch::TensorOptions doubleDeviceOptions()
{
    return torch::TensorOptions {}.dtype(torch::kFloat64).device(torch::kCUDA);
}

template<typename T>
inline torch::TensorOptions HostOptions();

template<typename T>
inline torch::TensorOptions DeviceOptions();

template<>
inline torch::TensorOptions HostOptions<int8_t>()
{
    return int8HostOptions();
}

template<>
inline torch::TensorOptions HostOptions<int16_t>()
{
    return int16HostOptions();
}

template<>
inline torch::TensorOptions HostOptions<int32_t>()
{
    return int32HostOptions();
}

template<>
inline torch::TensorOptions HostOptions<int64_t>()
{
    return int64HostOptions();
}

template<>
inline torch::TensorOptions HostOptions<float>()
{
    return floatHostOptions();
}

template<>
inline torch::TensorOptions HostOptions<double>()
{
    return doubleHostOptions();
}

template<>
inline torch::TensorOptions DeviceOptions<int8_t>()
{
    return int8DeviceOptions();
}

template<>
inline torch::TensorOptions DeviceOptions<int16_t>()
{
    return int16DeviceOptions();
}

template<>
inline torch::TensorOptions DeviceOptions<int32_t>()
{
    return int32DeviceOptions();
}

template<>
inline torch::TensorOptions DeviceOptions<int64_t>()
{
    return int64DeviceOptions();
}

template<>
inline torch::TensorOptions DeviceOptions<float>()
{
    return floatDeviceOptions();
}

template<>
inline torch::TensorOptions DeviceOptions<double>()
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
inline torch::Tensor
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
inline torch::Tensor createTensorFromPointer(void* pointer,
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
inline torch::Tensor createHostTensorFromPointer(T* pointer, const at::IntArrayRef& size)
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
inline torch::Tensor createHostTensorFromPointer(T* pointer, const at::IntArrayRef& size, const at::IntArrayRef& stride)
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
inline torch::Tensor createDeviceTensorFromPointer(T* pointer, const at::IntArrayRef& size)
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
inline torch::Tensor
createDeviceTensorFromPointer(T* pointer, const at::IntArrayRef& size, const at::IntArrayRef& stride)
{
    return createTensorFromPointer(pointer, size, stride, TensorOptions::DeviceOptions<T>());
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership.
 * This function maps the host pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, 15) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
inline torch::Tensor getVertexBufferAsHostTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getHostPointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createHostTensorFromPointer(vertex_buffer, {num_vertices, 15});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership.
 * This function maps the device pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, 15) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
inline torch::Tensor getVertexBufferAsDeviceTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getDevicePointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createDeviceTensorFromPointer(vertex_buffer, {num_vertices, 15});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex positions.
 * This function maps the host pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, 3) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
inline torch::Tensor getPositionsAsHostTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getHostPointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createHostTensorFromPointer(vertex_buffer, {num_vertices, 3}, {15, 1});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex positions.
 * This function maps the device pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, 3) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
inline torch::Tensor getPositionsAsDeviceTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getDevicePointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createDeviceTensorFromPointer(vertex_buffer, {num_vertices, 3}, {15, 1});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex colors.
 * This function maps the host pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, 3) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
inline torch::Tensor getColorsAsHostTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getHostPointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createHostTensorFromPointer(vertex_buffer + 3, {num_vertices, 3}, {15, 1});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex colors.
 * This function maps the device pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, 3) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
inline torch::Tensor getColorsAsDeviceTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getDevicePointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createDeviceTensorFromPointer(vertex_buffer + 3, {num_vertices, 3}, {15, 1});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex normals.
 * This function maps the host pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, 3) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
inline torch::Tensor getNormalsAsHostTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getHostPointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createHostTensorFromPointer(vertex_buffer + 6, {num_vertices, 3}, {15, 1});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex normals.
 * This function maps the device pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, 3) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
inline torch::Tensor getNormalsAsDeviceTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getDevicePointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createDeviceTensorFromPointer(vertex_buffer + 6, {num_vertices, 3}, {15, 1});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex tangents.
 * This function maps the host pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, 3) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
inline torch::Tensor getTangentsAsHostTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getHostPointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createHostTensorFromPointer(vertex_buffer + 9, {num_vertices, 3}, {15, 1});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex tangents.
 * This function maps the device pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, 3) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
inline torch::Tensor getTangentsAsDeviceTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getDevicePointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createDeviceTensorFromPointer(vertex_buffer + 9, {num_vertices, 3}, {15, 1});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex uvs.
 * This function maps the host pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, 3) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
inline torch::Tensor getUVsAsHostTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getHostPointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createHostTensorFromPointer(vertex_buffer + 12, {num_vertices, 3}, {15, 1});
}

/**
 * @brief Create a tensor from a vertex buffer without taking ownership. It creates a view to the vertex uvs.
 * This function maps the device pointer of the vertex buffer. It has to be unmapped manually by the caller if the given
 * vertex buffer is used when rendering. It is assumed that this vertex buffer represents the vertices in an atcg::Graph
 * object. Therefore, it will have a size of (n_vertices, 3) (see atcg::Vertex).
 *
 * @param buffer The vertex buffer
 *
 * @return A tensor that wraps around the data pointer
 */
inline torch::Tensor getUVsAsDeviceTensor(const atcg::ref_ptr<atcg::VertexBuffer>& buffer)
{
    float* vertex_buffer  = buffer->getDevicePointer<float>();
    uint32_t num_vertices = buffer->size() / sizeof(atcg::Vertex);
    return createDeviceTensorFromPointer(vertex_buffer + 12, {num_vertices, 3}, {15, 1});
}

}    // namespace atcg