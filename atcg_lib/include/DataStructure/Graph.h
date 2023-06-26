#pragma once

#include <vector>

#include <Core/Memory.h>
#include <Core/glm.h>
#include <Renderer/Buffer.h>
#include <Renderer/VertexArray.h>

namespace atcg
{

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
 * @brief A struct to model a vertex
 */
struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;
};

/**
 * @brief A struct to model an edge
 */
struct Edge
{
    glm::vec2 indices;
    glm::vec3 color;
    float radius;
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
     * @brief Create a point cloud.
     * The data gets directly uploaded to the GPU for rendering
     *
     * @param vertices The vertices of the point cloud
     *
     * @return The graph
     */
    static atcg::ref_ptr<Graph> createPointCloud(const std::vector<Vertex>& vertices);

    /**
     * @brief Create a triangle mesh.
     * The data gets directly uploaded to the GPU for rendering
     *
     * @param vertices The vertices
     * @param face_indices The faces
     * @param edge_radius The default radius of edges
     *
     * @return The graph
     */
    static atcg::ref_ptr<Graph> createTriangleMesh(const std::vector<Vertex>& vertices,
                                                   const std::vector<glm::u32vec3>& face_indices,
                                                   float edge_radius = 0.1f);

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
     * @brief Create a point cloud from a device buffer.
     * @note The pointer is assumed to be a device pointer. If ATCG_CUDA_BACKEND is not defined, it is assumed to be
     * a standard host pointer and a memcpy from host to device (OpenGL) is performed. If ATCG_CUDA_BACKEND is defined,
     * it is assumed to be a CUDA pointer and a memcpy from device (CUDA) to device (OpenGL) is performed.
     *
     * If num_vertices is smaller than capacity_vertices() the buffer will be reused without a new allocation.
     *
     * @param vertices The vertices
     * @param num_vertices The number of vertices
     */
    static atcg::ref_ptr<Graph> createPointCloud(const Vertex* vertices, uint32_t num_vertices);

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
     * @param num_vertices The number of vertices
     * @param indices The indices
     * @param num_faces The number of faces
     * @param edge_radius The default radius of the edges
     *
     * @return The graph
     */
    static atcg::ref_ptr<Graph> createTriangleMesh(const Vertex* vertices,
                                                   uint32_t num_vertices,
                                                   const glm::u32vec3* indices,
                                                   uint32_t num_faces,
                                                   float edge_radius = 0.1f);

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
     * @param num_vertices The number of vertices
     * @param indices The edges
     * @param num_faces The number of edges
     *
     * @return The graph
     */
    static atcg::ref_ptr<Graph>
    createGraph(const Vertex* vertices, uint32_t num_vertices, const Edge* edges, uint32_t num_edges);

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
     * If num_vertices is smaller than capacity_vertices() the buffer will be reused without a new allocation.
     *
     * @param vertices The new vertex information
     */
    void updateVertices(const std::vector<Vertex>& vertices);

    /**
     * @brief Update the faces.
     *
     * If num_faces is smaller than capacity_faces() the buffer will be reused without a new allocation.
     *
     * @param vertices The vertices of the new faces
     * @param faces The new face information
     */
    void updateFaces(const std::vector<Vertex>& vertices, const std::vector<glm::u32vec3>& faces);

    /**
     * @brief Update the edges.
     *
     * If num_edges is smaller than capacity_edges() the buffer will be reused without a new allocation.
     *
     * @param edges The new edge information
     */
    void updateEdges(const std::vector<Edge>& edges);

    /**
     * @brief Update the vertices.
     * @note The pointer is assumed to be a device pointer. If ATCG_CUDA_BACKEND is not defined, it is assumed to be
     * a standard host pointer and a memcpy from host to device (OpenGL) is performed. If ATCG_CUDA_BACKEND is defined,
     * it is assumed to be a CUDA pointer and a memcpy from device (CUDA) to device (OpenGL) is performed.
     *
     * If num_vertices is smaller than capacity_vertices() the buffer will be reused without a new allocation.
     *
     * @param vertices The new vertex information
     */
    void updateVertices(const Vertex* vertices, uint32_t num_vertices);

    /**
     * @brief Update the faces.
     * @note The pointer is assumed to be a device pointer. If ATCG_CUDA_BACKEND is not defined, it is assumed to be
     * a standard host pointer and a memcpy from host to device (OpenGL) is performed. If ATCG_CUDA_BACKEND is defined,
     * it is assumed to be a CUDA pointer and a memcpy from device (CUDA) to device (OpenGL) is performed.
     *
     * If num_faces is smaller than capacity_faces() the buffer will be reused without a new allocation.
     *
     * @param faces The new facce information
     */
    void updateFaces(const glm::u32vec3* faces, uint32_t num_faces);

    /**
     * @brief Update the edges.
     * @note The pointer is assumed to be a device pointer. If ATCG_CUDA_BACKEND is not defined, it is assumed to be
     * a standard host pointer and a memcpy from host to device (OpenGL) is performed. If ATCG_CUDA_BACKEND is defined,
     * it is assumed to be a CUDA pointer and a memcpy from device (CUDA) to device (OpenGL) is performed.
     *
     * If num_edges is smaller than capacity_edges() the buffer will be reused without a new allocation.
     *
     * @param edges The new edge information
     */
    void updateEdges(const Edge* edges, uint32_t num_edges);

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
     * @brief Get the number of vertices
     *
     * @return Number of vertices
     */
    uint32_t capacity_vertices() const;

    /**
     * @brief Get the number of edges
     *
     * @return Number of edges
     */
    uint32_t capacity_edges() const;

    /**
     * @brief Get the number of faces
     *
     * @return Number of faces
     */
    uint32_t capacity_faces() const;

    /**
     * @brief Get the type of the graph
     *
     * @return The type
     */
    GraphType type() const;

private:
    class Impl;
    atcg::scope_ptr<Impl> impl;
};

namespace IO
{
atcg::ref_ptr<Graph> read_mesh(const std::string& path);
}

}    // namespace atcg