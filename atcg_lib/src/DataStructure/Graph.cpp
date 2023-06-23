#include <DataStructure/Graph.h>

#pragma once

#include <vector>

#include <Core/Memory.h>
#include <Core/glm.h>
#include <Renderer/Buffer.h>
#include <Renderer/VertexArray.h>

namespace atcg
{

class Graph::Impl
{
public:
    Impl();

    ~Impl();

    void createVertexBuffer(const std::vector<Vertex>& vertices);
    void createEdgeBuffer(const std::vector<Edge>& edges);

    atcg::ref_ptr<VertexBuffer> vertices;
    atcg::ref_ptr<IndexBuffer> indices;
    atcg::ref_ptr<VertexBuffer> edges;

    atcg::ref_ptr<VertexArray> vertices_array;
    atcg::ref_ptr<VertexArray> edges_array;

    uint32_t n_vertices;
    uint32_t n_edges;
    uint32_t n_faces;

    GraphType type;
};

Graph::Impl::Impl() {}

Graph::Impl::~Impl() {}

void Graph::Impl::createVertexBuffer(const std::vector<Vertex>& vertices)
{
    this->vertices = atcg::make_ref<VertexBuffer>((void*)vertices.data(), vertices.size() * sizeof(Vertex));
    this->vertices->setLayout({{ShaderDataType::Float3, "aPosition"},
                         {ShaderDataType::Float3, "aNormal"},
                         {ShaderDataType::Float3, "aColor"}});


    vertices_array = atcg::make_ref<VertexArray>();
    vertices_array->pushVertexBuffer(this->vertices);
}

void Graph::Impl::createEdgeBuffer(const std::vector<Edge>& edges)
{
    this->edges = atcg::make_ref<VertexBuffer>((void*)edges.data(), edges.size() * sizeof(Edge));
    this->edges->setLayout(
        {{ShaderDataType::Float2, "aIndex"}, {ShaderDataType::Float3, "aColor"}, {ShaderDataType::Float, "aRadius"}});


    edges_array = atcg::make_ref<VertexArray>();
    edges_array->pushVertexBuffer(this->edges);
}

Graph::Graph()
{
    impl = atcg::make_scope<Impl>();
}

Graph::~Graph() {}

atcg::ref_ptr<Graph> Graph::createPointCloud(const std::vector<Vertex>& vertices)
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();
    result->impl->createVertexBuffer(vertices);
    result->impl->type = GraphType::ATCG_GRAPH_TYPE_POINTCLOUD;
    return result;
}

atcg::ref_ptr<Graph> Graph::createTriangleMesh(const std::vector<Vertex>& vertices,
                                               const std::vector<glm::u32vec3>& face_indices)
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();
    result->impl->createVertexBuffer(vertices);
    result->impl->indices = atcg::make_ref<IndexBuffer>((uint32_t*)face_indices.data(), face_indices.size() * 3);
    result->impl->vertices_array->setIndexBuffer(result->impl->indices);
    result->impl->n_faces = face_indices.size();
    result->impl->type    = GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH;
    // TODO create Edge buffer
    return result;
}

atcg::ref_ptr<Graph> Graph::createGraph(const std::vector<Vertex>& vertices, const std::vector<Edge>& edges)
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();
    result->impl->createVertexBuffer(vertices);
    result->impl->createEdgeBuffer(edges);
    result->impl->type = GraphType::ATCG_GRAPH_TYPE_GRAPH;
    return result;
}

atcg::ref_ptr<Graph> Graph::createPointCloud(const Vertex* vertices, uint32_t num_vertices)
{
    // TODO
    return nullptr;
}

atcg::ref_ptr<Graph> Graph::createTriangleMesh(const Vertex* vertices,
                                               const glm::u32vec3* indices,
                                               uint32_t num_vertices,
                                               uint32_t num_faces)
{
    // TODO
    return nullptr;
}

atcg::ref_ptr<Graph>
Graph::createGraph(const Vertex* vertices, const Edge* edges, uint32_t num_vertices, uint32_t num_edges)
{
    // TODO
    return nullptr;
}

const atcg::ref_ptr<VertexBuffer>& Graph::getVerticesBuffer() const
{
    return impl->vertices;
}

const atcg::ref_ptr<IndexBuffer>& Graph::getFaceIndices() const
{
    return impl->indices;
}

const atcg::ref_ptr<VertexBuffer>& Graph::getEdgeIndices() const
{
    return impl->edges;
}

const atcg::ref_ptr<VertexArray>& Graph::getVerticesArray() const
{
    return impl->vertices_array;
}

const atcg::ref_ptr<VertexArray>& Graph::getEdgesArray() const
{
    return impl->edges_array;
}

void Graph::updateVertices(const std::vector<Vertex>& vertices) {}

void Graph::updateFaces(const std::vector<glm::u32vec3>& faces) {}

void Graph::updateEdges(const std::vector<Edge>& edges) {}

void Graph::updateVertices(const Vertex* vertices, uint32_t num_vertices) {}

void Graph::updateFaces(const glm::u32vec3* faces, uint32_t num_faces) {}

void Graph::updateEdges(const Edge* edges, uint32_t num_edges) {}

uint32_t Graph::n_vertices() const
{
    return impl->n_vertices;
}

uint32_t Graph::n_edges() const
{
    return impl->n_edges;
}

uint32_t Graph::n_faces() const
{
    return impl->n_faces;
}

GraphType Graph::type() const
{
    return impl->type;
}
}    // namespace atcg