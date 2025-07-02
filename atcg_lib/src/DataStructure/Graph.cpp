#include <DataStructure/Graph.h>
#include <DataStructure/TorchUtils.h>

namespace atcg
{

using namespace torch::indexing;

class Graph::Impl
{
public:
    Impl();

    ~Impl();

    void initBuffers();

    void updateVertexBuffer(const torch::Tensor& vertices);
    void updateEdgeBuffer(const torch::Tensor& edges);
    void updateFaceBuffer(const torch::Tensor& indices);
    torch::Tensor edgesFromIndices(const torch::Tensor& face_indices);

    atcg::ref_ptr<VertexBuffer> vertices = nullptr;
    atcg::ref_ptr<IndexBuffer> indices   = nullptr;
    atcg::ref_ptr<VertexBuffer> edges    = nullptr;

    atcg::ref_ptr<VertexArray> vertices_array;
    atcg::ref_ptr<VertexArray> edges_array;

    uint32_t n_vertices = 0;
    uint32_t n_edges    = 0;
    uint32_t n_faces    = 0;

    GraphType type;
};

Graph::Impl::Impl()
{
    initBuffers();
}

Graph::Impl::~Impl() {}

void Graph::Impl::initBuffers()
{
    // Create necessary buffers always on init
    this->vertices = atcg::make_ref<VertexBuffer>();
    this->vertices->setLayout({{ShaderDataType::Float3, "aPosition"},
                               {ShaderDataType::Float3, "aColor"},
                               {ShaderDataType::Float3, "aNormal"},
                               {ShaderDataType::Float3, "aTangent"},
                               {ShaderDataType::Float3, "aUV"}});


    vertices_array = atcg::make_ref<VertexArray>();
    vertices_array->pushVertexBuffer(this->vertices);

    this->edges = atcg::make_ref<VertexBuffer>();
    this->edges->setLayout(
        {{ShaderDataType::Float2, "aIndex"}, {ShaderDataType::Float3, "aColor"}, {ShaderDataType::Float, "aRadius"}});


    edges_array = atcg::make_ref<VertexArray>();
    edges_array->pushVertexBuffer(this->edges);

    indices = atcg::make_ref<IndexBuffer>();
}

void Graph::Impl::updateVertexBuffer(const torch::Tensor& pvertices)
{
    uint32_t n = pvertices.size(0);
    if(n == 0) return;

    vertices->resize(n * sizeof(atcg::Vertex));
    if(pvertices.is_cuda())
    {
#ifdef ATCG_CUDA_BACKEND
        float* vertex_buffer_ptr = vertices->getDevicePointer<float>();

        CUDA_SAFE_CALL(cudaMemcpy((void*)vertex_buffer_ptr,
                                  pvertices.data_ptr(),
                                  sizeof(atcg::Vertex) * n,
                                  cudaMemcpyDeviceToDevice));

        vertices->unmapDevicePointers();
#else
        vertices->setData(pvertices.data_ptr(), sizeof(atcg::Vertex) * n);
#endif
    }
    else
    {
        vertices->setData(pvertices.data_ptr(), sizeof(atcg::Vertex) * n);
    }
    n_vertices = n;
}

void Graph::Impl::updateEdgeBuffer(const torch::Tensor& pedges)
{
    uint32_t n = pedges.size(0);
    if(n == 0) return;

    edges->resize(sizeof(atcg::Edge) * n);
    if(pedges.is_cuda())
    {
#ifdef ATCG_CUDA_BACKEND
        float* edge_buffer_ptr = edges->getDevicePointer<float>();

        CUDA_SAFE_CALL(
            cudaMemcpy((void*)edge_buffer_ptr, pedges.data_ptr(), sizeof(atcg::Edge) * n, cudaMemcpyDeviceToDevice));

        edges->unmapDevicePointers();
#else
        edges->setData(pedges.data_ptr(), sizeof(atcg::Edge) * n);
#endif
    }
    else
    {
        edges->setData(pedges.data_ptr(), sizeof(atcg::Edge) * n);
    }
    n_edges = n;
}

void Graph::Impl::updateFaceBuffer(const torch::Tensor& pindices)
{
    uint32_t n = pindices.size(0);
    if(n == 0) return;

    indices->resize(sizeof(glm::u32vec3) * n);
    if(pindices.is_cuda())
    {
#ifdef ATCG_CUDA_BACKEND
        uint32_t* face_buffer_ptr = indices->getDevicePointer<uint32_t>();

        CUDA_SAFE_CALL(cudaMemcpy((void*)face_buffer_ptr,
                                  pindices.data_ptr(),
                                  sizeof(glm::u32vec3) * n,
                                  cudaMemcpyDeviceToDevice));

        indices->unmapDevicePointers();
#else
        indices->setData((const uint32_t*)pindices.data_ptr(), 3 * n);
#endif
    }
    else
    {
        indices->setData((const uint32_t*)pindices.data_ptr(), 3 * n);
    }

    n_faces = n;
}

torch::Tensor Graph::Impl::edgesFromIndices(const torch::Tensor& indices)
{
    torch::Tensor e1 = indices.index({Slice(), Slice(0, 2)});
    torch::Tensor e2 = indices.index({Slice(), Slice(1, 3)});
    torch::Tensor e3 = indices.index({Slice(), Slice(0, 3, 2)});

    torch::Tensor edges = torch::vstack({e1, e2, e3}).cpu();
    edges               = std::get<0>(torch::sort(edges, 1));
    edges               = std::get<0>(torch::unique_dim(edges, 0, false));

    torch::Tensor edge_buffers = torch::ones({edges.size(0), 6});
    edge_buffers.index_put_({Slice(), Slice(0, 2)}, edges);

    return edge_buffers;
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

    result->updateVertices(vertices);

    result->impl->type = GraphType::ATCG_GRAPH_TYPE_POINTCLOUD;
    return result;
}

atcg::ref_ptr<Graph> Graph::createPointCloud(const atcg::MemoryBuffer<Vertex, device_allocator>& vertices)
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();

    result->updateVertices(vertices);

    result->impl->type = GraphType::ATCG_GRAPH_TYPE_POINTCLOUD;
    return result;
}

atcg::ref_ptr<Graph> Graph::createPointCloud(const torch::Tensor& vertices)
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();

    result->impl->updateVertexBuffer(vertices);

    result->impl->type = GraphType::ATCG_GRAPH_TYPE_POINTCLOUD;
    return result;
}

atcg::ref_ptr<Graph> Graph::createTriangleMesh()
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();

    result->impl->vertices_array->setIndexBuffer(result->impl->indices);
    result->impl->type = GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH;
    return result;
}


atcg::ref_ptr<Graph> Graph::createTriangleMesh(const std::vector<Vertex>& vertices,
                                               const std::vector<glm::u32vec3>& face_indices)
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();

    result->updateVertices(vertices);
    result->updateFaces(face_indices);
    result->impl->vertices_array->setIndexBuffer(result->impl->indices);

    result->impl->type = GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH;
    return result;
}

atcg::ref_ptr<Graph> Graph::createTriangleMesh(const atcg::MemoryBuffer<Vertex, device_allocator>& vertices,
                                               const atcg::MemoryBuffer<glm::u32vec3, device_allocator>& indices)
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();

    result->updateVertices(vertices);
    result->updateFaces(indices);
    result->impl->vertices_array->setIndexBuffer(result->impl->indices);

    result->impl->type = GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH;
    return result;
}

atcg::ref_ptr<Graph> Graph::createTriangleMesh(const torch::Tensor& vertices, const torch::Tensor& indices)
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();

    if(vertices.is_cuda() != indices.is_cuda())
    {
        ATCG_ERROR("Graph::createTriangleMesh: Vertices and Indices tensor not on same device!");
        return result;
    }

    result->updateVertices(vertices);
    result->updateFaces(indices);
    result->impl->vertices_array->setIndexBuffer(result->impl->indices);

    result->impl->type = GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH;
    return result;
}

atcg::ref_ptr<Graph> Graph::createTriangleMesh(const atcg::ref_ptr<TriMesh>& mesh)
{
    mesh->request_vertex_normals();
    mesh->request_face_normals();
    mesh->update_normals();

    std::vector<Vertex> vertex_data;
    vertex_data.resize(mesh->n_vertices());

    std::vector<glm::u32vec3> indices_data;
    indices_data.resize(mesh->n_faces());

    bool has_color     = mesh->has_vertex_colors();
    bool has_texcoords = mesh->has_vertex_texcoords2D();

    for(auto vertex = mesh->vertices_begin(); vertex != mesh->vertices_end(); ++vertex)
    {
        int32_t vertex_id               = vertex->idx();
        glm::vec3 pos                   = mesh->point(*vertex);
        glm::vec3 normal                = mesh->calc_vertex_normal(*vertex);
        glm::vec3 col                   = has_color ? mesh->color(*vertex) : glm::vec3(1);
        glm::vec2 uv                    = has_texcoords ? mesh->texcoord2D(*vertex) : glm::vec2(0);
        vertex_data[vertex_id].position = pos;
        vertex_data[vertex_id].normal   = normal;
        vertex_data[vertex_id].color    = has_color ? col / 255.0f : glm::vec3(1.0f);
        vertex_data[vertex_id].uv       = glm::vec3(uv, 0);
        vertex_data[vertex_id].tangent  = glm::vec3(0);    // TODO
    }

    int32_t face_id = 0;
    for(auto face = mesh->faces_begin(); face != mesh->faces_end(); ++face)
    {
        int32_t vertex_id = 0;
        int32_t vertices[3];    // Local copy of vertex indices for tangent calculation
        for(auto vertex = face->vertices().begin(); vertex != face->vertices().end(); ++vertex)
        {
            glm::value_ptr(indices_data[face_id])[vertex_id] = vertex->idx();
            vertices[vertex_id]                              = vertex->idx();
            ++vertex_id;
        }

        // Calculate tangent vector
        glm::vec3 pos1 = vertex_data[vertices[0]].position;
        glm::vec3 pos2 = vertex_data[vertices[1]].position;
        glm::vec3 pos3 = vertex_data[vertices[2]].position;

        glm::vec2 uv1 = vertex_data[vertices[0]].uv;
        glm::vec2 uv2 = vertex_data[vertices[1]].uv;
        glm::vec2 uv3 = vertex_data[vertices[2]].uv;

        glm::vec3 edge1 = pos2 - pos1;
        glm::vec3 edge2 = pos3 - pos1;

        glm::vec2 deltaUV1 = uv2 - uv1;
        glm::vec2 deltaUV2 = uv3 - uv1;

        float det = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

        glm::vec3 tangent;
        tangent.x = det * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
        tangent.y = det * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
        tangent.z = det * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
        tangent   = glm::normalize(tangent);

        vertex_data[vertices[0]].tangent += tangent;
        vertex_data[vertices[1]].tangent += tangent;
        vertex_data[vertices[2]].tangent += tangent;

        ++face_id;
    }

    // TODO: Find better structure so we don't have to iterate over everything again
    for(auto& vtx: vertex_data)
    {
        vtx.tangent = glm::normalize(vtx.tangent);
    }

    return createTriangleMesh(vertex_data, indices_data);
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

    result->updateVertices(vertices);
    result->updateEdges(edges);

    result->impl->type = GraphType::ATCG_GRAPH_TYPE_GRAPH;
    return result;
}

atcg::ref_ptr<Graph> Graph::createGraph(const atcg::MemoryBuffer<Vertex, device_allocator>& vertices,
                                        const atcg::MemoryBuffer<Edge, device_allocator>& edges)
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();

    result->updateVertices(vertices);
    result->updateEdges(edges);

    result->impl->type = GraphType::ATCG_GRAPH_TYPE_GRAPH;
    return result;
}

atcg::ref_ptr<Graph> Graph::createGraph(const torch::Tensor& vertices, const torch::Tensor& edges)
{
    atcg::ref_ptr<Graph> result = atcg::make_ref<Graph>();

    if(vertices.is_cuda() != edges.is_cuda())
    {
        ATCG_ERROR("Graph::createGraph: Vertices and Edges tensor not on same device!");
        return result;
    }

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
    torch::Tensor tvertices = atcg::createHostTensorFromPointer((float*)vertices.data(), {(int)vertices.size(), 15});
    impl->updateVertexBuffer(tvertices);
}

void Graph::updateVertices(const atcg::MemoryBuffer<Vertex, device_allocator>& vertices)
{
#ifdef ATCG_CUDA_BACKEND
    torch::Tensor tvertices = atcg::createDeviceTensorFromPointer((float*)vertices.get(), {(int)vertices.size(), 15});
#else
    torch::Tensor tvertices = atcg::createHostTensorFromPointer((float*)vertices.get(), {(int)vertices.size(), 15});
#endif

    impl->updateVertexBuffer(tvertices);
}

void Graph::updateVertices(const torch::Tensor& vertices)
{
    impl->updateVertexBuffer(vertices);
}

void Graph::updateFaces(const std::vector<glm::u32vec3>& faces)
{
    torch::Tensor tindices = atcg::createHostTensorFromPointer((uint32_t*)faces.data(), {(int)faces.size(), 3});

    updateFaces(tindices);
}

void Graph::updateFaces(const atcg::MemoryBuffer<glm::u32vec3, atcg::device_allocator>& faces)
{
#ifdef ATCG_CUDA_BACKEND
    torch::Tensor tindices = atcg::createDeviceTensorFromPointer((uint32_t*)faces.get(), {(int)faces.size(), 3});
#else
    torch::Tensor tindices = atcg::createHostTensorFromPointer((uint32_t*)faces.get(), {(int)faces.size(), 3});
#endif

    updateFaces(tindices);
}

void Graph::updateFaces(const torch::Tensor& faces)
{
    impl->updateFaceBuffer(faces);
    torch::Tensor edges = impl->edgesFromIndices(faces);
    impl->updateEdgeBuffer(edges);
}

void Graph::updateEdges(const std::vector<Edge>& edges)
{
    torch::Tensor tedges = atcg::createHostTensorFromPointer((float*)edges.data(), {(int)edges.size(), 6});

    impl->updateEdgeBuffer(tedges);
}

void Graph::updateEdges(const atcg::MemoryBuffer<Edge, device_allocator>& edges)
{
#ifdef ATCG_CUDA_BACKEND
    torch::Tensor tedges = atcg::createDeviceTensorFromPointer((float*)edges.get(), {(int)edges.size(), 6});
#else
    torch::Tensor tedges = atcg::createHostTensorFromPointer((float*)edges.get(), {(int)edges.size(), 6});
#endif

    impl->updateEdgeBuffer(tedges);
}

void Graph::updateEdges(const torch::Tensor& edges)
{
    impl->updateEdgeBuffer(edges);
}

void Graph::resizeVertices(uint32_t size)
{
    impl->vertices->resize(size * sizeof(atcg::Vertex));
    impl->n_vertices = size;
}

void Graph::resizeFaces(uint32_t size)
{
    impl->indices->resize(size * sizeof(glm::u32vec3));
    impl->n_faces = size;
}

void Graph::resizeEdges(uint32_t size)
{
    impl->edges->resize(size * sizeof(atcg::Edge));
    impl->n_edges = size;
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

torch::Tensor Graph::getPositions(const torch::Device& device) const
{
    if(device.is_cpu())
    {
        return atcg::getPositionsAsHostTensor(impl->vertices);
    }
    else
    {
        return atcg::getPositionsAsDeviceTensor(impl->vertices);
    }
}

torch::Tensor Graph::getColors(const torch::Device& device) const
{
    if(device.is_cpu())
    {
        return atcg::getColorsAsHostTensor(impl->vertices);
    }
    else
    {
        return atcg::getColorsAsDeviceTensor(impl->vertices);
    }
}

torch::Tensor Graph::getNormals(const torch::Device& device) const
{
    if(device.is_cpu())
    {
        return atcg::getNormalsAsHostTensor(impl->vertices);
    }
    else
    {
        return atcg::getNormalsAsDeviceTensor(impl->vertices);
    }
}

torch::Tensor Graph::getTangents(const torch::Device& device) const
{
    if(device.is_cpu())
    {
        return atcg::getTangentsAsHostTensor(impl->vertices);
    }
    else
    {
        return atcg::getTangentsAsDeviceTensor(impl->vertices);
    }
}

torch::Tensor Graph::getUVs(const torch::Device& device) const
{
    if(device.is_cpu())
    {
        return atcg::getUVsAsHostTensor(impl->vertices);
    }
    else
    {
        return atcg::getUVsAsDeviceTensor(impl->vertices);
    }
}

torch::Tensor Graph::getEdges(const torch::Device& device) const
{
    if(device.is_cpu())
    {
        float* edge_pointer = impl->edges->getHostPointer<float>();
        return atcg::createHostTensorFromPointer<float>(edge_pointer, {n_edges(), 6});
    }
    else
    {
        float* edge_pointer = impl->edges->getDevicePointer<float>();
        return atcg::createDeviceTensorFromPointer<float>(edge_pointer, {n_edges(), 6});
    }
}

torch::Tensor Graph::getFaces(const torch::Device& device) const
{
    if(device.is_cpu())
    {
        uint32_t* face_pointer = impl->indices->getHostPointer<uint32_t>();
        return atcg::createHostTensorFromPointer<uint32_t>(face_pointer, {n_faces(), 3});
    }
    else
    {
        uint32_t* face_pointer = impl->indices->getDevicePointer<uint32_t>();
        return atcg::createDeviceTensorFromPointer<uint32_t>(face_pointer, {n_faces(), 3});
    }
}

void Graph::unmapHostVertexPointer()
{
    impl->vertices->unmapHostPointers();
}

void Graph::unmapDeviceVertexPointer()
{
    impl->vertices->unmapDevicePointers();
}

void Graph::unmapVertexPointer()
{
    impl->vertices->unmapPointers();
}

void Graph::unmapHostEdgePointer()
{
    impl->edges->unmapHostPointers();
}

void Graph::unmapDeviceEdgePointer()
{
    impl->edges->unmapDevicePointers();
}

void Graph::unmapEdgePointer()
{
    impl->edges->unmapPointers();
}

void Graph::unmapHostFacePointer()
{
    impl->indices->unmapHostPointers();
}

void Graph::unmapDeviceFacePointer()
{
    impl->indices->unmapDevicePointers();
}

void Graph::unmapFacePointer()
{
    impl->indices->unmapPointers();
}

void Graph::unmapAllHostPointers()
{
    unmapHostVertexPointer();
    unmapHostEdgePointer();
    unmapHostFacePointer();
}

void Graph::unmapAllDevicePointers()
{
    unmapDeviceVertexPointer();
    unmapDeviceEdgePointer();
    unmapDeviceFacePointer();
}

void Graph::unmapAllPointers()
{
    unmapVertexPointer();
    unmapEdgePointer();
    unmapFacePointer();
}

atcg::ref_ptr<Graph> Graph::copy() const
{
    atcg::ref_ptr<Graph> result;

    switch(impl->type)
    {
        case atcg::GraphType::ATCG_GRAPH_TYPE_GRAPH:
        {
            result = Graph::createGraph(atcg::getVertexBufferAsDeviceTensor(impl->vertices), getDeviceEdges());
        }
        break;
        case atcg::GraphType::ATCG_GRAPH_TYPE_POINTCLOUD:
        {
            result = Graph::createPointCloud(atcg::getVertexBufferAsDeviceTensor(impl->vertices));
        }
        break;
        case atcg::GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH:
        {
            result = Graph::createTriangleMesh(atcg::getVertexBufferAsDeviceTensor(impl->vertices), getDeviceFaces());
        }
        break;
    }

    impl->vertices->unmapDevicePointers();
    impl->edges->unmapDevicePointers();
    impl->indices->unmapDevicePointers();

    return result;
}

void Graph::copy(const atcg::ref_ptr<Graph>& other)
{
    unmapAllPointers();
    impl->type = other->impl->type;
    impl->initBuffers();
    handle = other->handle;

    switch(impl->type)
    {
        case GraphType::ATCG_GRAPH_TYPE_GRAPH:
        {
            updateVertices(atcg::getVertexBufferAsDeviceTensor(other->getVerticesBuffer()));
            updateEdges(other->getDeviceEdges());
        }
        break;
        case GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH:
        {
            updateVertices(atcg::getVertexBufferAsDeviceTensor(other->getVerticesBuffer()));
            updateFaces(other->getDeviceFaces());
            impl->vertices_array->setIndexBuffer(impl->indices);
        }
        break;
        case GraphType::ATCG_GRAPH_TYPE_POINTCLOUD:
        {
            updateVertices(atcg::getVertexBufferAsDeviceTensor(other->getVerticesBuffer()));
        }
        break;
    }

    other->unmapAllPointers();
}

}    // namespace atcg