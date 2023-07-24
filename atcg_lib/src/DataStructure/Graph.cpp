#include <DataStructure/Graph.h>

#include <unordered_set>

namespace atcg
{

struct Vec2Hasher
{
    std::size_t operator()(const glm::vec2& v) const
    {
        return (*reinterpret_cast<const uint32_t*>(&v.x) * 73856093) ^
               (*reinterpret_cast<const uint32_t*>(&v.y) * 19349669);
    }
};

class Graph::Impl
{
public:
    Impl();

    ~Impl();

    void createVertexBuffer(const Vertex* vertices, uint32_t num_vertices);
    void createEdgeBuffer(const Edge* edges, uint32_t num_edges);
    void createFaceBuffer(const glm::u32vec3* face_indices, uint32_t num_faces);
    std::vector<Edge> edgesFromIndices(const std::vector<glm::u32vec3>& face_indices);

    atcg::ref_ptr<VertexBuffer> vertices = nullptr;
    atcg::ref_ptr<IndexBuffer> indices   = nullptr;
    atcg::ref_ptr<VertexBuffer> edges    = nullptr;

    atcg::ref_ptr<VertexArray> vertices_array;
    atcg::ref_ptr<VertexArray> edges_array;

    uint32_t n_vertices = 0;
    uint32_t n_edges    = 0;
    uint32_t n_faces    = 0;

    float edge_radius = 0.1f;

    GraphType type;
};

Graph::Impl::Impl() {}

Graph::Impl::~Impl() {}

void Graph::Impl::createVertexBuffer(const Vertex* vertices, uint32_t num_vertices)
{
    if(num_vertices == 0) return;

    if(n_vertices == 0)
    {
        this->vertices = atcg::make_ref<VertexBuffer>((void*)vertices, num_vertices * sizeof(Vertex));
        this->vertices->setLayout({{ShaderDataType::Float3, "aPosition"},
                                   {ShaderDataType::Float3, "aNormal"},
                                   {ShaderDataType::Float3, "aColor"}});


        vertices_array = atcg::make_ref<VertexArray>();
        vertices_array->pushVertexBuffer(this->vertices);

        n_vertices = num_vertices;
    }
    else
    {
        this->vertices->setData((void*)vertices, sizeof(Vertex) * num_vertices);
        n_vertices = num_vertices;
    }
}

void Graph::Impl::createEdgeBuffer(const Edge* edges, uint32_t num_edges)
{
    if(num_edges == 0) return;

    if(n_edges == 0)
    {
        this->edges = atcg::make_ref<VertexBuffer>((void*)edges, num_edges * sizeof(Edge));
        this->edges->setLayout({{ShaderDataType::Float2, "aIndex"},
                                {ShaderDataType::Float3, "aColor"},
                                {ShaderDataType::Float, "aRadius"}});


        edges_array = atcg::make_ref<VertexArray>();
        edges_array->pushVertexBuffer(this->edges);

        n_edges = num_edges;
    }
    else
    {
        this->edges->setData((void*)edges, sizeof(Edge) * num_edges);
        n_edges = num_edges;
    }
}

void Graph::Impl::createFaceBuffer(const glm::u32vec3* face_indices, uint32_t num_faces)
{
    if(num_faces == 0) return;
    if(n_faces == 0)
    {
        indices = atcg::make_ref<IndexBuffer>((uint32_t*)face_indices, num_faces * 3);
        vertices_array->setIndexBuffer(indices);

        n_faces = num_faces;
    }
    else
    {
        indices->setData((uint32_t*)face_indices, num_faces * 3);
        n_faces = num_faces;
    }
}

std::vector<Edge> Graph::Impl::edgesFromIndices(const std::vector<glm::u32vec3>& face_indices)
{
    std::unordered_set<glm::vec2, Vec2Hasher> edge_set;

    std::vector<Edge> edge_buffer;
    for(glm::u32vec3 triangle: face_indices)
    {
        uint32_t v1 = triangle.x;
        uint32_t v2 = triangle.y;
        uint32_t v3 = triangle.z;

        // glm::vec3 color_v1 = vertices[v1].color;
        // glm::vec3 color_v2 = vertices[v2].color;
        // glm::vec3 color_v3 = vertices[v3].color;

        glm::vec2 edges[3] = {glm::vec2(std::min(v1, v2), std::max(v1, v2)),
                              glm::vec2(std::min(v2, v3), std::max(v2, v3)),
                              glm::vec2(std::min(v3, v1), std::max(v3, v1))};

        // glm::vec3 colors[3] = {glm::mix(color_v1, color_v2, 0.5f),
        //                        glm::mix(color_v2, color_v3, 0.5f),
        //                        glm::mix(color_v3, color_v1, 0.5f)};


        for(uint32_t i = 0; i < 3; ++i)
        {
            if(edge_set.find(edges[i]) == edge_set.end())
            {
                edge_set.insert(edges[i]);
                edge_buffer.push_back({edges[i], glm::vec3(1), edge_radius});
            }
        }
    }

    return edge_buffer;
}

Graph::Graph()
{
    impl = atcg::make_scope<Impl>();
}

Graph::~Graph() {}

atcg::ref_ptr<Graph> Graph::createPointCloud()
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();

    result->impl->type = GraphType::ATCG_GRAPH_TYPE_POINTCLOUD;
    return result;
}

atcg::ref_ptr<Graph> Graph::createPointCloud(const std::vector<Vertex>& vertices)
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();
    result->impl->createVertexBuffer(vertices.data(), vertices.size());

    result->impl->type = GraphType::ATCG_GRAPH_TYPE_POINTCLOUD;
    return result;
}

atcg::ref_ptr<Graph> Graph::createTriangleMesh(float edge_radius)
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();
    result->impl->edge_radius   = edge_radius;

    result->impl->type = GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH;
    return result;
}


atcg::ref_ptr<Graph> Graph::createTriangleMesh(const std::vector<Vertex>& vertices,
                                               const std::vector<glm::u32vec3>& face_indices,
                                               float edge_radius)
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();
    result->impl->createVertexBuffer(vertices.data(), vertices.size());
    result->impl->createFaceBuffer(face_indices.data(), face_indices.size());
    result->impl->edge_radius     = edge_radius;
    std::vector<Edge> edge_buffer = result->impl->edgesFromIndices(face_indices);
    result->impl->createEdgeBuffer(edge_buffer.data(), edge_buffer.size());

    result->impl->type = GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH;
    return result;
}

atcg::ref_ptr<Graph> Graph::createTriangleMesh(const atcg::ref_ptr<TriMesh>& mesh, float edge_radius)
{
    mesh->request_vertex_normals();
    mesh->request_face_normals();
    mesh->update_normals();

    std::vector<Vertex> vertex_data;
    vertex_data.resize(mesh->n_vertices());

    std::vector<glm::u32vec3> indices_data;
    indices_data.resize(mesh->n_faces());

    bool has_color = mesh->has_vertex_colors();

    for(auto vertex = mesh->vertices_begin(); vertex != mesh->vertices_end(); ++vertex)
    {
        int32_t vertex_id               = vertex->idx();
        glm::vec3 pos                   = mesh->point(*vertex);
        glm::vec3 normal                = mesh->calc_vertex_normal(*vertex);
        glm::vec3 col                   = has_color ? mesh->color(*vertex) : glm::vec3(1);
        vertex_data[vertex_id].position = pos;
        vertex_data[vertex_id].normal   = normal;
        vertex_data[vertex_id].color    = has_color ? col / 255.0f : glm::vec3(1.0f);
    }

    int32_t face_id = 0;
    for(auto face = mesh->faces_begin(); face != mesh->faces_end(); ++face)
    {
        int32_t vertex_id = 0;
        for(auto vertex = face->vertices().begin(); vertex != face->vertices().end(); ++vertex)
        {
            glm::value_ptr(indices_data[face_id])[vertex_id] = vertex->idx();
            ++vertex_id;
        }
        ++face_id;
    }

    return createTriangleMesh(vertex_data, indices_data, edge_radius);
}

atcg::ref_ptr<Graph> Graph::createGraph()
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();

    result->impl->type = GraphType::ATCG_GRAPH_TYPE_GRAPH;
    return result;
}

atcg::ref_ptr<Graph> Graph::createGraph(const std::vector<Vertex>& vertices, const std::vector<Edge>& edges)
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();
    result->impl->createVertexBuffer(vertices.data(), vertices.size());
    result->impl->createEdgeBuffer(edges.data(), edges.size());

    result->impl->type = GraphType::ATCG_GRAPH_TYPE_GRAPH;
    return result;
}

atcg::ref_ptr<Graph> Graph::createPointCloud(const atcg::ref_ptr<Vertex, device_allocator>& vertices)
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();
    result->updateVertices(vertices);

    result->impl->type = GraphType::ATCG_GRAPH_TYPE_POINTCLOUD;
    return result;
}

// atcg::ref_ptr<Graph> Graph::createTriangleMesh(const Vertex* vertices,
//                                                uint32_t num_vertices,
//                                                const glm::u32vec3* indices,
//                                                uint32_t num_faces,
//                                                float edge_radius)
// {
//     // TODO
//     return nullptr;
// }

atcg::ref_ptr<Graph> Graph::createGraph(const atcg::ref_ptr<Vertex, device_allocator>& vertices,
                                        const atcg::ref_ptr<Edge, device_allocator>& edges)
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();
    result->updateVertices(vertices);
    result->updateEdges(edges);

    result->impl->type = GraphType::ATCG_GRAPH_TYPE_GRAPH;
    return result;
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

void Graph::updateVertices(const std::vector<Vertex>& vertices)
{
    impl->createVertexBuffer(vertices.data(), vertices.size());
}

void Graph::updateFaces(const std::vector<glm::u32vec3>& faces)
{
    impl->createFaceBuffer(faces.data(), faces.size());
    std::vector<Edge> edges = impl->edgesFromIndices(faces);
    impl->createEdgeBuffer(edges.data(), edges.size());
}

void Graph::updateEdges(const std::vector<Edge>& edges)
{
    impl->createEdgeBuffer(edges.data(), edges.size());
}

void Graph::updateVertices(const atcg::ref_ptr<Vertex, device_allocator>& vertices)
{
    impl->createVertexBuffer(nullptr, vertices.size());

#ifdef ATCG_CUDA_BACKEND
    bool mapped   = impl->vertices->isDeviceMapped();
    void* dev_ptr = impl->vertices->getDevicePointer();
    CUDA_SAFE_CALL(
        cudaMemcpy(dev_ptr, (void*)vertices.get(), sizeof(Vertex) * vertices.size(), cudaMemcpyDeviceToDevice));
    if(!mapped) { impl->vertices->unmapDevicePointers(); }
#else
    impl->vertices->setData(vertices.get(), vertices.size() * sizeof(Vertex));
#endif
}

// void Graph::updateFaces(const glm::u32vec3* faces, uint32_t num_faces) {}

void Graph::updateEdges(const atcg::ref_ptr<Edge, device_allocator>& edges)
{
    impl->createEdgeBuffer(nullptr, edges.size());

#ifdef ATCG_CUDA_BACKEND
    bool mapped   = impl->edges->isDeviceMapped();
    void* dev_ptr = impl->edges->getDevicePointer();
    CUDA_SAFE_CALL(cudaMemcpy(dev_ptr, (void*)edges.get(), sizeof(Edge) * edges.size(), cudaMemcpyDeviceToDevice));
    if(!mapped) { impl->edges->unmapDevicePointers(); }
#else
    impl->edges->setData(edges.get(), edges.size() * sizeof(Edge));
#endif
}

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

float Graph::edge_radius() const
{
    return impl->edge_radius;
}

atcg::ref_ptr<Graph> IO::read_mesh(const std::string& path, OpenMesh::IO::Options options)
{
    // TODO: Replace this with dedicated obj loader
    atcg::ref_ptr<TriMesh> mesh = atcg::make_ref<TriMesh>();
    mesh->request_vertex_normals();
    mesh->request_face_normals();

    if(options.vertex_has_color()) { mesh->request_vertex_colors(); }
    if(options.face_has_color()) { mesh->request_face_colors(); }

    OpenMesh::IO::read_mesh(*mesh.get(), path, options);

    return Graph::createTriangleMesh(mesh, 0.01f);
}

}    // namespace atcg