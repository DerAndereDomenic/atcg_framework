#include <DataStructure/Graph.h>

#include <unordered_set>

// TEMPORARY
#include <OpenMesh/OpenMesh.h>

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

    uint32_t n_vertices = 0;
    uint32_t n_edges    = 0;
    uint32_t n_faces    = 0;

    uint32_t cap_vertices = 0;
    uint32_t cap_edges    = 0;
    uint32_t cap_faces    = 0;

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
    result->impl->type         = GraphType::ATCG_GRAPH_TYPE_POINTCLOUD;
    result->impl->n_vertices   = vertices.size();
    result->impl->cap_vertices = vertices.size();
    return result;
}

struct Vec2Hasher
{
    std::size_t operator()(const glm::vec2& v) const
    {
        return (*reinterpret_cast<const uint32_t*>(&v.x) * 73856093) ^
               (*reinterpret_cast<const uint32_t*>(&v.y) * 19349669);
    }
};

atcg::ref_ptr<Graph> Graph::createTriangleMesh(const std::vector<Vertex>& vertices,
                                               const std::vector<glm::u32vec3>& face_indices,
                                               float edge_radius)
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();
    result->impl->createVertexBuffer(vertices);
    result->impl->indices = atcg::make_ref<IndexBuffer>((uint32_t*)face_indices.data(), face_indices.size() * 3);
    result->impl->vertices_array->setIndexBuffer(result->impl->indices);
    result->impl->type = GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH;

    std::unordered_set<glm::vec2, Vec2Hasher> edge_set;

    std::vector<Edge> edge_buffer;
    for(glm::u32vec3 triangle: face_indices)
    {
        uint32_t v1 = triangle.x;
        uint32_t v2 = triangle.y;
        uint32_t v3 = triangle.z;

        glm::vec3 color_v1 = vertices[v1].color;
        glm::vec3 color_v2 = vertices[v2].color;
        glm::vec3 color_v3 = vertices[v3].color;

        glm::vec2 edges[3] = {glm::vec2(std::min(v1, v2), std::max(v1, v2)),
                              glm::vec2(std::min(v2, v3), std::max(v2, v3)),
                              glm::vec2(std::min(v3, v1), std::max(v3, v1))};

        glm::vec3 colors[3] = {glm::mix(color_v1, color_v2, 0.5f),
                               glm::mix(color_v2, color_v3, 0.5f),
                               glm::mix(color_v3, color_v1, 0.5f)};

        for(uint32_t i = 0; i < 3; ++i)
        {
            if(edge_set.find(edges[i]) == edge_set.end())
            {
                edge_set.insert(edges[i]);
                edge_buffer.push_back({edges[i], colors[i], edge_radius});
            }
        }
    }

    result->impl->createEdgeBuffer(edge_buffer);

    result->impl->n_faces    = face_indices.size();
    result->impl->n_vertices = vertices.size();
    result->impl->n_edges    = edge_buffer.size();

    result->impl->cap_faces    = face_indices.size();
    result->impl->cap_vertices = vertices.size();
    result->impl->cap_edges    = edge_buffer.size();

    return result;
}

atcg::ref_ptr<Graph> Graph::createGraph(const std::vector<Vertex>& vertices, const std::vector<Edge>& edges)
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();
    result->impl->createVertexBuffer(vertices);
    result->impl->createEdgeBuffer(edges);
    result->impl->type         = GraphType::ATCG_GRAPH_TYPE_GRAPH;
    result->impl->n_vertices   = vertices.size();
    result->impl->n_edges      = edges.size();
    result->impl->cap_vertices = vertices.size();
    result->impl->cap_edges    = edges.size();
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

const atcg::ref_ptr<IndexBuffer>& Graph::getFaceIndexBuffer() const
{
    return impl->indices;
}

const atcg::ref_ptr<VertexBuffer>& Graph::getEdgesBuffer() const
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

uint32_t Graph::capacity_vertices() const
{
    return impl->n_vertices;
}

uint32_t Graph::capacity_edges() const
{
    return impl->n_edges;
}

uint32_t Graph::capacity_faces() const
{
    return impl->n_faces;
}

GraphType Graph::type() const
{
    return impl->type;
}

atcg::ref_ptr<Graph> IO::read_mesh(const std::string& path)
{
    // TODO: Replace this with dedicated obj loader
    TriMesh mesh;
    OpenMesh::IO::read_mesh(mesh, path);

    mesh.request_vertex_normals();
    mesh.request_face_normals();
    mesh.update_normals();

    std::vector<Vertex> vertex_data;
    vertex_data.resize(mesh.n_vertices());

    std::vector<glm::u32vec3> indices_data;
    indices_data.resize(mesh.n_faces());

    bool has_color = mesh.has_vertex_colors();

    for(auto vertex = mesh.vertices_begin(); vertex != mesh.vertices_end(); ++vertex)
    {
        int32_t vertex_id               = vertex->idx();
        glm::vec3 pos                   = mesh.point(*vertex);
        glm::vec3 normal                = mesh.calc_vertex_normal(*vertex);
        glm::vec3 col                   = has_color ? mesh.color(*vertex) : glm::vec3(1);
        vertex_data[vertex_id].position = pos;
        vertex_data[vertex_id].normal   = normal;
        vertex_data[vertex_id].color    = has_color ? col / 255.0f : glm::vec3(1.0f);
    }

    int32_t face_id = 0;
    for(auto face = mesh.faces_begin(); face != mesh.faces_end(); ++face)
    {
        int32_t vertex_id = 0;
        for(auto vertex = face->vertices().begin(); vertex != face->vertices().end(); ++vertex)
        {
            glm::value_ptr(indices_data[face_id])[vertex_id] = vertex->idx();
            ++vertex_id;
        }
        ++face_id;
    }

    return Graph::createTriangleMesh(vertex_data, indices_data, 0.01f);
}

}    // namespace atcg