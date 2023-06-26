#pragma once

#include <vector>

#include <Core/Memory.h>
#include <Core/glm.h>
#include <Renderer/Buffer.h>
#include <Renderer/VertexArray.h>

namespace atcg
{
enum class GraphType
{
    ATCG_GRAPH_TYPE_POINTCLOUD,
    ATCG_GRAPH_TYPE_TRIANGLEMESH,
    ATCG_GRAPH_TYPE_GRAPH
};

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;
};

struct Edge
{
    glm::vec2 indices;
    glm::vec3 color;
    float radius;
};

class Graph
{
public:
    Graph();

    ~Graph();

    static atcg::ref_ptr<Graph> createPointCloud(const std::vector<Vertex>& vertices);

    static atcg::ref_ptr<Graph> createTriangleMesh(const std::vector<Vertex>& vertices,
                                                   const std::vector<glm::u32vec3>& face_indices);

    static atcg::ref_ptr<Graph> createGraph(const std::vector<Vertex>& vertices, const std::vector<Edge>& edges);

    static atcg::ref_ptr<Graph> createPointCloud(const Vertex* vertices, uint32_t num_vertices);

    static atcg::ref_ptr<Graph>
    createTriangleMesh(const Vertex* vertices, const glm::u32vec3* indices, uint32_t num_vertices, uint32_t num_faces);

    static atcg::ref_ptr<Graph>
    createGraph(const Vertex* vertices, const Edge* edges, uint32_t num_vertices, uint32_t num_edges);

    const atcg::ref_ptr<VertexBuffer>& getVerticesBuffer() const;

    const atcg::ref_ptr<IndexBuffer>& getFaceIndices() const;

    const atcg::ref_ptr<VertexBuffer>& getEdgeIndices() const;

    const atcg::ref_ptr<VertexArray>& getVerticesArray() const;

    const atcg::ref_ptr<VertexArray>& getEdgesArray() const;

    void updateVertices(const std::vector<Vertex>& vertices);

    void updateFaces(const std::vector<glm::u32vec3>& faces);

    void updateEdges(const std::vector<Edge>& edges);

    void updateVertices(const Vertex* vertices, uint32_t num_vertices);

    void updateFaces(const glm::u32vec3* faces, uint32_t num_faces);

    void updateEdges(const Edge* edges, uint32_t num_edges);

    uint32_t n_vertices() const;

    uint32_t n_edges() const;

    uint32_t n_faces() const;

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