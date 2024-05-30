#pragma once


#include <Core/Memory.h>
#include <Core/glm.h>
#include <Renderer/Buffer.h>
#include <Renderer/VertexArray.h>

#include <OpenMesh/OpenMesh.h>
#include <vector>
#include <DataStructure/TorchUtils.h>
#include <DataStructure/GraphDefinitions.h>

namespace atcg
{
class Scene;

/**
 * @brief An enum to distinguish between different graph types
 */
enum class GraphType
{
    ATCG_GRAPH_TYPE_POINTCLOUD,
    ATCG_GRAPH_TYPE_TRIANGLEMESH,
    ATCG_GRAPH_TYPE_GRAPH
};

/**
 * @brief A structure to model different geometries
 */
class Graph
{
public:
    /**
     * @brief Constructor
     */
    Graph();

    /**
     * @brief Destructor
     */
    ~Graph();

    /**
     * @brief Create an empty point cloud
     * @return The graph
     */
    static atcg::ref_ptr<Graph> createPointCloud();

    /**
     * @brief Create a point cloud.
     * The data gets directly uploaded to the GPU for rendering
     *
     * @param vertices The vertices of the point cloud
     *
     * @return The graph
     */
    static atcg::ref_ptr<Graph> createPointCloud(const std::vector<Vertex>& vertices);

    /**
     * @brief Create a point cloud from a device buffer.
     * @note The pointer is assumed to be a device pointer. If ATCG_CUDA_BACKEND is not defined, it is assumed to be
     * a standard host pointer and a memcpy from host to device (OpenGL) is performed. If ATCG_CUDA_BACKEND is defined,
     * it is assumed to be a CUDA pointer and a memcpy from device (CUDA) to device (OpenGL) is performed.
     *
     * If num_vertices is smaller than capacity_vertices() the buffer will be reused without a new allocation.
     *
     * @param vertices The vertices
     *
     * @return The graph
     */
    static atcg::ref_ptr<Graph> createPointCloud(const atcg::MemoryBuffer<Vertex, device_allocator>& vertices);

    /**
     * @brief Create a point cloud from a tensor.
     * The tensor can contain either device or host memory.
     *
     * If num_vertices is smaller than capacity_vertices() the buffer will be reused without a new allocation.
     *
     * @param vertices The vertices
     *
     * @return The graph
     *
     */
    static atcg::ref_ptr<Graph> createPointCloud(const torch::Tensor& vertices);

    /**
     * @brief Create an empty mesh
     *
     * @return The graph
     */
    static atcg::ref_ptr<Graph> createTriangleMesh();

    /**
     * @brief Create a triangle mesh.
     * The data gets directly uploaded to the GPU for rendering
     *
     * @param vertices The vertices
     * @param face_indices The faces
     *
     * @return The graph
     */
    static atcg::ref_ptr<Graph> createTriangleMesh(const std::vector<Vertex>& vertices,
                                                   const std::vector<glm::u32vec3>& face_indices);

    /**
     * @brief Create a triangle mesh from a device buffer.
     * @note The pointer is assumed to be a device pointer. If ATCG_CUDA_BACKEND is not defined, it is assumed to be
     * a standard host pointer and a memcpy from host to device (OpenGL) is performed. If ATCG_CUDA_BACKEND is defined,
     * it is assumed to be a CUDA pointer and a memcpy from device (CUDA) to device (OpenGL) is performed.
     *
     * If num_vertices is smaller than capacity_vertices() the buffer will be reused without a new allocation.
     * If num_faces is smaller than capacity_faces() the buffer will be reused without a new allocation.
     *
     * @param vertices The vertices
     * @param indices The indices
     *
     * @return The graph
     */
    static atcg::ref_ptr<Graph> createTriangleMesh(const atcg::MemoryBuffer<Vertex, device_allocator>& vertices,
                                                   const atcg::MemoryBuffer<glm::u32vec3, device_allocator>& indices);

    /**
     * @brief Create a triangle mesh from a tensor.
     * The tensor can contain either device or host memory.
     *
     * If num_vertices is smaller than capacity_vertices() the buffer will be reused without a new allocation.
     * If num_faces is smaller than capacity_faces() the buffer will be reused without a new allocation.
     *
     * @param vertices The vertices
     * @param indices The face indices
     *
     * @return The graph
     *
     */
    static atcg::ref_ptr<Graph> createTriangleMesh(const torch::Tensor& vertices, const torch::Tensor& indices);

    /**
     * @brief Create a triangle mesh from a half edge structure.
     * The data gets directly uploaded to the GPU for rendering
     *
     * @param trimesh The halfedge structure
     *
     * @return The graph
     */
    static atcg::ref_ptr<Graph> createTriangleMesh(const atcg::ref_ptr<TriMesh>& trimesh);

    /**
     * @brief Create an empty graph
     * @return The graph
     */
    static atcg::ref_ptr<Graph> createGraph();

    /**
     * @brief Create a graph.
     * The data gets directly uploaded to the GPU for rendering.
     * It does not hold any face information for now. It only connects vertices by (arbitrary) edges.
     *
     * @param vertices The vertices
     * @param Edge The edges
     *
     * @return The graph
     */
    static atcg::ref_ptr<Graph> createGraph(const std::vector<Vertex>& vertices, const std::vector<Edge>& edges);

    /**
     * @brief Create a graph mesh from a device buffer.
     * @note The pointer is assumed to be a device pointer. If ATCG_CUDA_BACKEND is not defined, it is assumed to be
     * a standard host pointer and a memcpy from host to device (OpenGL) is performed. If ATCG_CUDA_BACKEND is defined,
     * it is assumed to be a CUDA pointer and a memcpy from device (CUDA) to device (OpenGL) is performed.
     *
     * If num_vertices is smaller than capacity_vertices() the buffer will be reused without a new allocation.
     * If num_edges is smaller than capacity_edges() the buffer will be reused without a new allocation.
     *
     * @param vertices The vertices
     * @param edges The edges
     *
     * @return The graph
     */
    static atcg::ref_ptr<Graph> createGraph(const atcg::MemoryBuffer<Vertex, device_allocator>& vertices,
                                            const atcg::MemoryBuffer<Edge, device_allocator>& edges);

    /**
     * @brief Create a graph from a tensor.
     * The tensor can contain either device or host memory.
     *
     * If num_vertices is smaller than capacity_vertices() the buffer will be reused without a new allocation.
     * If num_edges is smaller than capacity_edges() the buffer will be reused without a new allocation.
     *
     * @param vertices The vertices
     * @param edges The edges
     *
     * @return The graph
     *
     */
    static atcg::ref_ptr<Graph> createGraph(const torch::Tensor& vertices, const torch::Tensor& edges);

    /**
     * @brief Get the vertex buffer that stores the vertex information
     *
     * @return The vertex buffer
     */
    const atcg::ref_ptr<VertexBuffer>& getVerticesBuffer() const;

    /**
     * @brief Get the index buffer that stores the triangle indices
     *
     * @return The index buffer
     */
    const atcg::ref_ptr<IndexBuffer>& getFaceIndexBuffer() const;

    /**
     * @brief Get the vertex buffer that stores the edge information
     *
     * @return The edge buffer
     */
    const atcg::ref_ptr<VertexBuffer>& getEdgesBuffer() const;

    /**
     * @brief Get the vertex array that holds layout information about the vertex buffer of the vertices
     *
     * @return The vertex array
     */
    const atcg::ref_ptr<VertexArray>& getVerticesArray() const;

    /**
     * @brief Get the vertex array that holds layout information about the vertex buffer of the edges
     *
     * @return The vertex array
     */
    const atcg::ref_ptr<VertexArray>& getEdgesArray() const;

    /**
     * @brief Update the vertices.
     *
     * If num_vertices is smaller than vertices.capacity() the buffer will be reused without a new allocation.
     *
     * @param vertices The new vertex information
     */
    void updateVertices(const std::vector<Vertex>& vertices);

    /**
     * @brief Update the vertices.
     * @note The pointer is assumed to be a device pointer. If ATCG_CUDA_BACKEND is not defined, it is assumed to be
     * a standard host pointer and a memcpy from host to device (OpenGL) is performed. If ATCG_CUDA_BACKEND is defined,
     * it is assumed to be a CUDA pointer and a memcpy from device (CUDA) to device (OpenGL) is performed.
     *
     * If num_vertices is smaller than vertices.capacity() the buffer will be reused without a new allocation.
     *
     * @param vertices The new vertex information
     */
    void updateVertices(const atcg::MemoryBuffer<Vertex, device_allocator>& vertices);

    /**
     * @brief Update the vertices.
     * The tensor can contain either device or host memory.
     *
     * If num_vertices is smaller than vertices.capacity() the buffer will be reused without a new allocation.
     *
     * @param vertices The new vertex information
     */
    void updateVertices(const torch::Tensor& vertices);

    /**
     * @brief Update the faces.
     *
     * If num_faces is smaller than faces.capacity() the buffer will be reused without a new allocation.
     *
     * @param vertices The vertices of the new faces
     * @param faces The new face information
     */
    void updateFaces(const std::vector<glm::u32vec3>& faces);

    /**
     * @brief Update the faces.
     * @note The pointer is assumed to be a device pointer. If ATCG_CUDA_BACKEND is not defined, it is assumed to be
     * a standard host pointer and a memcpy from host to device (OpenGL) is performed. If ATCG_CUDA_BACKEND is defined,
     * it is assumed to be a CUDA pointer and a memcpy from device (CUDA) to device (OpenGL) is performed.
     *
     * If num_faces is smaller than faces.capacity() the buffer will be reused without a new allocation.
     *
     * @param faces The new face information
     */
    void updateFaces(const atcg::MemoryBuffer<glm::u32vec3, atcg::device_allocator>& faces);

    /**
     * @brief Update the faces.
     * The tensor can contain host or device memory
     *
     * If num_faces is smaller than faces.capacity() the buffer will be reused without a new allocation.
     *
     * @param faces The new face information
     */
    void updateFaces(const torch::Tensor& faces);

    /**
     * @brief Update the edges.
     *
     * If num_edges is smaller than edges.capacity() the buffer will be reused without a new allocation.
     *
     * @param edges The new edge information
     */
    void updateEdges(const std::vector<Edge>& edges);

    /**
     * @brief Update the edges.
     * @note The pointer is assumed to be a device pointer. If ATCG_CUDA_BACKEND is not defined, it is assumed to be
     * a standard host pointer and a memcpy from host to device (OpenGL) is performed. If ATCG_CUDA_BACKEND is defined,
     * it is assumed to be a CUDA pointer and a memcpy from device (CUDA) to device (OpenGL) is performed.
     *
     * If num_edges is smaller than edges.capacity() the buffer will be reused without a new allocation.
     *
     * @param edges The new edge information
     */
    void updateEdges(const atcg::MemoryBuffer<Edge, device_allocator>& edges);

    /**
     * @brief Update the edges.
     * The tensor can contain host or device memory
     *
     * If num_edges is smaller than edges.capacity() the buffer will be reused without a new allocation.
     *
     * @param edges The new edge information
     */
    void updateEdges(const torch::Tensor& edges);

    /**
     * @brief Update the number of vertices.
     * Old data is lost if the new size is greater than the capacity of the vertex buffer.
     *
     * @param size The new size
     */
    void resizeVertices(uint32_t size);

    /**
     * @brief Update the number of faces.
     * Old data is lost if the new size is greater than the capacity of the vertex buffer.
     *
     * @param size The new size
     */
    void resizeFaces(uint32_t size);

    /**
     * @brief Update the number of edges.
     * Old data is lost if the new size is greater than the capacity of the vertex buffer.
     *
     * @param size The new size
     */
    void resizeEdges(uint32_t size);

    /**
     * @brief Get the number of vertices
     *
     * @return Number of vertices
     */
    uint32_t n_vertices() const;

    /**
     * @brief Get the number of edges
     *
     * @return Number of edges
     */
    uint32_t n_edges() const;

    /**
     * @brief Get the number of faces
     *
     * @return Number of faces
     */
    uint32_t n_faces() const;

    /**
     * @brief Get the type of the graph
     *
     * @return The type
     */
    GraphType type() const;

    /**
     * @brief Get a tensor of vertex positions on the specified device.
     * @note This function maps the OpenGL buffer to a pointer. It has to be unmapped manually by the caller if the
     * vertex buffer is used in any OpenGL operations afterward.
     * Changes in the tensor directly translate to changes to the internal mesh structure and affect the rendering.
     * Memory is non-contiguous.
     *
     * @param device The device
     *
     * @return A tensor that points to the vertex buffer.
     */
    torch::Tensor getPositions(const torch::Device& device) const;

    /**
     * @brief Get a tensor of vertex colors on the specified device.
     * @note This function maps the OpenGL buffer to a pointer. It has to be unmapped manually by the caller if the
     * vertex buffer is used in any OpenGL operations afterward.
     * Changes in the tensor directly translate to changes to the internal mesh structure and affect the rendering.
     * Memory is non-contiguous.
     *
     * @param device The device
     *
     * @return A tensor that points to the vertex buffer.
     */
    torch::Tensor getColors(const torch::Device& device) const;

    /**
     * @brief Get a tensor of vertex normals on the specified device.
     * @note This function maps the OpenGL buffer to a pointer. It has to be unmapped manually by the caller if the
     * vertex buffer is used in any OpenGL operations afterward.
     * Changes in the tensor directly translate to changes to the internal mesh structure and affect the rendering.
     * Memory is non-contiguous.
     *
     * @param device The device
     *
     * @return A tensor that points to the vertex buffer.
     */
    torch::Tensor getNormals(const torch::Device& device) const;

    /**
     * @brief Get a tensor of vertex tangents on the specified device.
     * @note This function maps the OpenGL buffer to a pointer. It has to be unmapped manually by the caller if the
     * vertex buffer is used in any OpenGL operations afterward.
     * Changes in the tensor directly translate to changes to the internal mesh structure and affect the rendering.
     * Memory is non-contiguous.
     *
     * @param device The device
     *
     * @return A tensor that points to the vertex buffer.
     */
    torch::Tensor getTangents(const torch::Device& device) const;

    /**
     * @brief Get a tensor of vertex uvs on the specified device.
     * @note This function maps the OpenGL buffer to a pointer. It has to be unmapped manually by the caller if the
     * vertex buffer is used in any OpenGL operations afterward.
     * Changes in the tensor directly translate to changes to the internal mesh structure and affect the rendering.
     * Memory is non-contiguous.
     *
     * @param device The device
     *
     * @return A tensor that points to the vertex buffer.
     */
    torch::Tensor getUVs(const torch::Device& device) const;

    /**
     * @brief Get a tensor of edges the specified device. The data format is specified by atcg::Edge (i.e., (idx1, idx2,
     * vec3(color), radius))
     * @note This function maps the OpenGL buffer to a pointer. It has to be unmapped manually by the caller if the
     * vertex buffer is used in any OpenGL operations afterward.
     * Changes in the tensor directly translate to changes to the internal mesh structure and affect the rendering.
     * Memory is non-contiguous.
     *
     * @param device The device
     *
     * @return A tensor that points to the vertex buffer.
     */
    torch::Tensor getEdges(const torch::Device& device) const;

    /**
     * @brief Get a tensor of face indices on the specified device.
     * @note This function maps the OpenGL buffer to a pointer. It has to be unmapped manually by the caller if the
     * vertex buffer is used in any OpenGL operations afterward.
     * Changes in the tensor directly translate to changes to the internal mesh structure and affect the rendering.
     * Memory is non-contiguous.
     *
     * @param device The device
     *
     * @return A tensor that points to the vertex buffer.
     */
    torch::Tensor getFaces(const torch::Device& device) const;

    /**
     * @brief Get the vertex positions as host tensor.
     * @note This function maps the host pointer of the undelying buffer. It has to be unmapped manually by the caller
     * before the buffer is used for any OpenGL calls. The tensor does not own the memory and is only valid as long as
     * the underlying VertexBuffer is valid. The memory is non-contiguous.
     *
     * @return The Tensor
     */
    ATCG_INLINE torch::Tensor getHostPositions() const { return getPositions(torch::Device(atcg::CPU)); }

    /**
     * @brief Get the vertex positions as device tensor.
     * @note This function maps the device pointer of the undelying buffer. It has to be unmapped manually by the caller
     * before the buffer is used for any OpenGL calls. The tensor does not own the memory and is only valid as long as
     * the underlying VertexBuffer is valid. The memory is non-contiguous.
     *
     * @return The Tensor
     */
    ATCG_INLINE torch::Tensor getDevicePositions() const { return getPositions(torch::Device(atcg::GPU)); }

    /**
     * @brief Get the vertex colors as host tensor.
     * @note This function maps the host pointer of the undelying buffer. It has to be unmapped manually by the caller
     * before the buffer is used for any OpenGL calls. The tensor does not own the memory and is only valid as long as
     * the underlying VertexBuffer is valid. The memory is non-contiguous.
     *
     * @return The Tensor
     */
    ATCG_INLINE torch::Tensor getHostColors() const { return getColors(torch::Device(atcg::CPU)); }

    /**
     * @brief Get the vertex colors as device tensor.
     * @note This function maps the device pointer of the undelying buffer. It has to be unmapped manually by the caller
     * before the buffer is used for any OpenGL calls. The tensor does not own the memory and is only valid as long as
     * the underlying VertexBuffer is valid. The memory is non-contiguous.
     *
     * @return The Tensor
     */
    ATCG_INLINE torch::Tensor getDeviceColors() const { return getColors(torch::Device(atcg::GPU)); }

    /**
     * @brief Get the vertex normals as host tensor.
     * @note This function maps the host pointer of the undelying buffer. It has to be unmapped manually by the caller
     * before the buffer is used for any OpenGL calls. The tensor does not own the memory and is only valid as long as
     * the underlying VertexBuffer is valid. The memory is non-contiguous.
     *
     * @return The Tensor
     */
    ATCG_INLINE torch::Tensor getHostNormals() const { return getNormals(torch::Device(atcg::CPU)); }

    /**
     * @brief Get the vertex normals as device tensor.
     * @note This function maps the device pointer of the undelying buffer. It has to be unmapped manually by the caller
     * before the buffer is used for any OpenGL calls. The tensor does not own the memory and is only valid as long as
     * the underlying VertexBuffer is valid. The memory is non-contiguous.
     *
     * @return The Tensor
     */
    ATCG_INLINE torch::Tensor getDeviceNormals() const { return getNormals(torch::Device(atcg::GPU)); }

    /**
     * @brief Get the vertex tangents as host tensor.
     * @note This function maps the host pointer of the undelying buffer. It has to be unmapped manually by the caller
     * before the buffer is used for any OpenGL calls. The tensor does not own the memory and is only valid as long as
     * the underlying VertexBuffer is valid. The memory is non-contiguous.
     *
     * @return The Tensor
     */
    ATCG_INLINE torch::Tensor getHostTangents() const { return getTangents(torch::Device(atcg::CPU)); }

    /**
     * @brief Get the vertex tangents as device tensor.
     * @note This function maps the device pointer of the undelying buffer. It has to be unmapped manually by the caller
     * before the buffer is used for any OpenGL calls. The tensor does not own the memory and is only valid as long as
     * the underlying VertexBuffer is valid. The memory is non-contiguous.
     *
     * @return The Tensor
     */
    ATCG_INLINE torch::Tensor getDeviceTangents() const { return getTangents(torch::Device(atcg::GPU)); }

    /**
     * @brief Get the vertex uvs as host tensor.
     * @note This function maps the host pointer of the undelying buffer. It has to be unmapped manually by the caller
     * before the buffer is used for any OpenGL calls. The tensor does not own the memory and is only valid as long as
     * the underlying VertexBuffer is valid. The memory is non-contiguous.
     *
     * @return The Tensor
     */
    ATCG_INLINE torch::Tensor getHostUVs() const { return getUVs(torch::Device(atcg::CPU)); }

    /**
     * @brief Get the vertex uvs as device tensor.
     * @note This function maps the device pointer of the undelying buffer. It has to be unmapped manually by the caller
     * before the buffer is used for any OpenGL calls. The tensor does not own the memory and is only valid as long as
     * the underlying VertexBuffer is valid. The memory is non-contiguous.
     *
     * @return The Tensor
     */
    ATCG_INLINE torch::Tensor getDeviceUVs() const { return getUVs(torch::Device(atcg::GPU)); }

    /**
     * @brief Get a tensor of edges the host. The data format is specified by atcg::Edge (i.e., (idx1, idx2,
     * vec3(color), radius))
     * @note This function maps the OpenGL buffer to a pointer. It has to be unmapped manually by the caller if the
     * vertex buffer is used in any OpenGL operations afterward.
     * Changes in the tensor directly translate to changes to the internal mesh structure and affect the rendering.
     * Memory is non-contiguous.
     *
     * @param device The device
     *
     * @return A tensor that points to the vertex buffer.
     */
    ATCG_INLINE torch::Tensor getHostEdges() const { return getEdges(torch::Device(atcg::CPU)); }

    /**
     * @brief Get a tensor of edges the host. The data format is specified by atcg::Edge (i.e., (idx1, idx2,
     * vec3(color), radius))
     * @note This function maps the OpenGL buffer to a pointer. It has to be unmapped manually by the caller if the
     * vertex buffer is used in any OpenGL operations afterward.
     * Changes in the tensor directly translate to changes to the internal mesh structure and affect the rendering.
     * Memory is non-contiguous.
     *
     * @param device The device
     *
     * @return A tensor that points to the vertex buffer.
     */
    ATCG_INLINE torch::Tensor getDeviceEdges() const { return getEdges(torch::Device(atcg::GPU)); }

    /**
     * @brief Get a tensor of face indices on the host.
     * @note This function maps the OpenGL buffer to a pointer. It has to be unmapped manually by the caller if the
     * vertex buffer is used in any OpenGL operations afterward.
     * Changes in the tensor directly translate to changes to the internal mesh structure and affect the rendering.
     * Memory is non-contiguous.
     *
     * @param device The device
     *
     * @return A tensor that points to the vertex buffer.
     */
    ATCG_INLINE torch::Tensor getHostFaces() const { return getFaces(torch::Device(atcg::CPU)); }

    /**
     * @brief Get a tensor of face indices on the device.
     * @note This function maps the OpenGL buffer to a pointer. It has to be unmapped manually by the caller if the
     * vertex buffer is used in any OpenGL operations afterward.
     * Changes in the tensor directly translate to changes to the internal mesh structure and affect the rendering.
     * Memory is non-contiguous.
     *
     * @param device The device
     *
     * @return A tensor that points to the vertex buffer.
     */
    ATCG_INLINE torch::Tensor getDeviceFaces() const { return getFaces(torch::Device(atcg::GPU)); }

    /**
     * @brief Unmaps all host pointers of the vertex buffer
     */
    void unmapHostVertexPointer();

    /**
     * @brief Unmaps all device pointers of the vertex buffer
     */
    void unmapDeviceVertexPointer();

    /**
     * @brief Unmaps all pointers of the vertex buffer
     */
    void unmapVertexPointer();

    /**
     * @brief Unmaps all host pointers of the edge buffer
     */
    void unmapHostEdgePointer();

    /**
     * @brief Unmaps all device pointers of the edge buffer
     */
    void unmapDeviceEdgePointer();

    /**
     * @brief Unmaps all pointers of the edge buffer
     */
    void unmapEdgePointer();

    /**
     * @brief Unmaps all host pointers of the face buffer
     */
    void unmapHostFacePointer();

    /**
     * @brief Unmaps all device pointers of the face buffer
     */
    void unmapDeviceFacePointer();

    /**
     * @brief Unmaps all pointers of the face buffer
     */
    void unmapFacePointer();

    /**
     * @brief Unmap all host pointers of opengl buffers (vertex, edge and face buffers)
     */
    void unmapAllHostPointers();

    /**
     * @brief Unmap all devicepointers of opengl buffers (vertex, edge and face buffers)
     */
    void unmapAllDevicePointers();

    /**
     * @brief Unmap all pointers of opengl buffers (vertex, edge and face buffers)
     */
    void unmapAllPointers();

private:
    class Impl;
    atcg::scope_ptr<Impl> impl;
};

namespace IO
{

/**
 * @brief Read an obj (triangle) mesh.
 * If the obj file contains multiple objects, they are all combined into one mesh representation.
 * Material files are ignored
 *
 * @param path The path to the obj file
 *
 * @return The mesh representation
 */
atcg::ref_ptr<Graph> read_mesh(const std::string& path);

/**
 * @brief Read an obj triangle cloud.
 * This method expects shapes that are created using the obj p tag. Face information is ignored.
 * If the obj file contains multiple objects, they are all combined into one point cloud representation.
 * Material files are ignored.
 *
 * @param path The path to the obj file
 *
 * @return The point cloud representation
 */
atcg::ref_ptr<Graph> read_pointcloud(const std::string& path);

/**
 * @brief Read an obj file representing a line collection.
 * This method ignores all surface models (triangle meshes) and only creates geometry from the line tag (l).
 * Only line segments with two vertices are supported (i.e. l x y).
 * Material files are ignored
 *
 * @param path The path to the obj file
 *
 * @return The line representation
 */
atcg::ref_ptr<Graph> read_lines(const std::string& path);

/**
 * @brief Read a scene representation from an obj file.
 * Create a scene object where each object is either a triangle mesh (f tag), a point cloud (p tag) or a line
 * collection. To use material files, export the obj with the pbr extension. The following parameters are used:
 *
 * - Kd - diffuse base color
 * - Pr - surface roughness
 * - Pm - surface metallic
 * - map_Kd - diffuse base color texture
 * - map_Pr - roughness texture
 * - map_Pm - metallic texture
 * - map_Bump - normal texture
 *
 * @note We do not use per-face materials but only per object materials
 *
 * @param path The path to the obj file
 *
 * @return The line representation
 */
atcg::ref_ptr<Scene> read_scene(const std::string& path);
}    // namespace IO

}    // namespace atcg