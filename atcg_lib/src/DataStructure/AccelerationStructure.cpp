#include <Pathtracing/AccelerationStructure.h>

namespace atcg
{
BVHAccelerationStructure::BVHAccelerationStructure(const atcg::ref_ptr<Graph>& mesh)
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

BVHAccelerationStructure::~BVHAccelerationStructure() {}
}    // namespace atcg