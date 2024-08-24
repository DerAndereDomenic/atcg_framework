#include <Pathtracing/AccelerationStructure.h>
#include <Scene/Components.h>
#include <Scene/Entity.h>
#include <Math/Utils.h>

namespace atcg
{
GASAccelerationStructure::GASAccelerationStructure(const atcg::ref_ptr<Graph>& mesh)
{
    if(mesh->type() != GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH)
    {
        ATCG_WARN("Can only create BVH for triangles. Aborting...");
        return;
    }

    bool vertices_mapped = mesh->getVerticesBuffer()->isHostMapped();
    bool faces_mapped    = mesh->getFaceIndexBuffer()->isHostMapped();

    _positions = mesh->getHostPositions().clone();
    _normals   = mesh->getHostNormals().clone();
    _uvs       = mesh->getHostUVs().clone();
    _faces     = mesh->getHostFaces().clone();

    // Restore original mapping relation
    if(!vertices_mapped)
    {
        mesh->getVerticesBuffer()->unmapHostPointers();
    }
    if(!faces_mapped)
    {
        mesh->getFaceIndexBuffer()->unmapHostPointers();
    }

    nanort::TriangleMesh<float> triangle_mesh(reinterpret_cast<const float*>(_positions.data_ptr()),
                                              reinterpret_cast<const uint32_t*>(_faces.data_ptr()),
                                              sizeof(float) * 3);
    nanort::TriangleSAHPred<float> triangle_pred(reinterpret_cast<const float*>(_positions.data_ptr()),
                                                 reinterpret_cast<const uint32_t*>(_faces.data_ptr()),
                                                 sizeof(float) * 3);
    bool ret = _bvh.Build(mesh->n_faces(), triangle_mesh, triangle_pred);
    assert(ret);

    nanort::BVHBuildStatistics stats = _bvh.GetStatistics();

    ATCG_INFO("BVH statistics:");
    ATCG_INFO("\t# of leaf   nodes: {0}", stats.num_leaf_nodes);
    ATCG_INFO("\t# of branch nodes: {0}", stats.num_branch_nodes);
    ATCG_INFO("\tMax tree depth   : {0}", stats.max_tree_depth);
}

GASAccelerationStructure::~GASAccelerationStructure() {}

void GASAccelerationStructure::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                                  const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    auto ptx_file = "C:/Users/Domenic/Documents/Repositories/atcg_framework/bin/Debug/MeshShape.dll";
    _hit_group    = pipeline->addTrianglesHitGroupShader({ptx_file, "__closesthit__mesh"}, {});
}

IASAccelerationStructure::IASAccelerationStructure(const atcg::ref_ptr<Scene>& scene)
{
    auto view = scene->getAllEntitiesWith<GeometryComponent, MeshRenderComponent, TransformComponent>();

    uint32_t total_vertices = 0;
    uint32_t total_faces    = 0;
    std::vector<atcg::ref_ptr<Graph>> geometry;
    std::vector<entt::entity> entity_handles;
    std::vector<atcg::TransformComponent> transforms;
    for(auto e: view)
    {
        Entity entity(e, scene.get());

        TransformComponent transform = entity.getComponent<TransformComponent>();

        _transforms.push_back(transform.getModel());
        transforms.push_back(transform);

        atcg::ref_ptr<Graph> graph = entity.getComponent<GeometryComponent>().graph;
        geometry.push_back(graph);

        _offsets.push_back(total_faces);
        total_vertices += graph->n_vertices();
        total_faces += graph->n_faces();
    }

    _positions = torch::empty({total_vertices, 3}, atcg::TensorOptions::floatHostOptions());
    _normals   = torch::empty({total_vertices, 3}, atcg::TensorOptions::floatHostOptions());
    _uvs       = torch::empty({total_vertices, 3}, atcg::TensorOptions::floatHostOptions());
    _faces     = torch::empty({total_faces, 3}, atcg::TensorOptions::int32HostOptions());
    _mesh_idx  = torch::empty({total_faces}, atcg::TensorOptions::int32HostOptions());

    uint32_t vertex_idx = 0;
    uint32_t face_idx   = 0;

    for(int i = 0; i < geometry.size(); ++i)
    {
        auto& graph = geometry[i];

        TransformComponent& transform = transforms[i];
        auto positions                = graph->getHostPositions().clone();
        auto normals                  = graph->getHostNormals().clone();
        auto uvs                      = graph->getHostUVs().clone();
        auto tangents                 = graph->getHostTangents().clone();
        auto faces                    = graph->getHostFaces().clone();
        applyTransform(positions, normals, tangents, transform);

        _positions.index_put_(
            {torch::indexing::Slice(vertex_idx, vertex_idx + graph->n_vertices()), torch::indexing::Slice()},
            positions);
        _normals.index_put_(
            {torch::indexing::Slice(vertex_idx, vertex_idx + graph->n_vertices()), torch::indexing::Slice()},
            normals);
        _uvs.index_put_(
            {torch::indexing::Slice(vertex_idx, vertex_idx + graph->n_vertices()), torch::indexing::Slice()},
            uvs);
        _faces.index_put_({torch::indexing::Slice(face_idx, face_idx + graph->n_faces()), torch::indexing::Slice()},
                          faces + (int32_t)vertex_idx);
        _mesh_idx.index_put_({torch::indexing::Slice(face_idx, face_idx + graph->n_faces())}, i);
        vertex_idx += graph->n_vertices();
        face_idx += graph->n_faces();
    }

    _positions.contiguous();
    _normals.contiguous();
    _uvs.contiguous();
    _faces.contiguous();
    _mesh_idx.contiguous();

    float* p_positions = _positions.data_ptr<float>();
    int32_t* p_faces   = _faces.data_ptr<int32_t>();
    nanort::TriangleMesh<float> triangle_mesh(p_positions, (uint32_t*)p_faces, sizeof(glm::vec3));
    nanort::TriangleSAHPred<float> triangle_pred(p_positions, (uint32_t*)p_faces, sizeof(glm::vec3));
    bool ret = _bvh.Build(total_faces, triangle_mesh, triangle_pred);

    nanort::BVHBuildStatistics stats = _bvh.GetStatistics();

    ATCG_INFO("BVH statistics:");
    ATCG_INFO("\t# of leaf   nodes: {0}", stats.num_leaf_nodes);
    ATCG_INFO("\t# of branch nodes: {0}", stats.num_branch_nodes);
    ATCG_INFO("\tMax tree depth   : {0}", stats.max_tree_depth);
}

IASAccelerationStructure::~IASAccelerationStructure() {}
}    // namespace atcg