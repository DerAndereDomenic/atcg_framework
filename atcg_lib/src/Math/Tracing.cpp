#include <Math/Tracing.h>
#include <Scene/Components.h>

namespace atcg
{

void Tracing::prepareAccelerationStructure(Entity entity)
{
    if(!entity.hasComponent<GeometryComponent>())
    {
        ATCG_WARN("Entity does not have a geometry component. Cancel BVH build...");
        return;
    }

    if(!entity.hasComponent<AccelerationStructureComponent>())
    {
        entity.addComponent<AccelerationStructureComponent>();
    }

    auto& acc_component = entity.getComponent<AccelerationStructureComponent>();

    auto& geometry_component  = entity.getComponent<GeometryComponent>();
    atcg::ref_ptr<Graph> mesh = geometry_component.graph;
    if(mesh->type() != GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH)
    {
        ATCG_WARN("Can only create BVH for triangles. Aborting...");
        return;
    }

    bool vertices_mapped = mesh->getVerticesBuffer()->isHostMapped();
    bool faces_mapped    = mesh->getFaceIndexBuffer()->isHostMapped();

    Vertex* vertices    = mesh->getVerticesBuffer()->getHostPointer<Vertex>();
    glm::u32vec3* faces = mesh->getFaceIndexBuffer()->getHostPointer<glm::u32vec3>();

    acc_component.vertices = atcg::MemoryBuffer<glm::vec3>(mesh->n_vertices());
    acc_component.faces    = atcg::MemoryBuffer<glm::u32vec3>(mesh->n_faces());

    std::vector<glm::vec3> temp_vertices(mesh->n_vertices());
    for(uint32_t i = 0; i < mesh->n_vertices(); ++i)
    {
        temp_vertices[i] = vertices[i].position;
    }
    acc_component.vertices.copy(temp_vertices.data());
    acc_component.faces.copy(faces);

    // Restore original mapping relation
    if(!vertices_mapped)
    {
        mesh->getVerticesBuffer()->unmapHostPointers();
    }
    if(!faces_mapped)
    {
        mesh->getFaceIndexBuffer()->unmapHostPointers();
    }

    nanort::TriangleMesh<float> triangle_mesh(reinterpret_cast<const float*>(acc_component.vertices.get()),
                                              reinterpret_cast<const uint32_t*>(acc_component.faces.get()),
                                              sizeof(float) * 3);
    nanort::TriangleSAHPred<float> triangle_pred(reinterpret_cast<const float*>(acc_component.vertices.get()),
                                                 reinterpret_cast<const uint32_t*>(acc_component.faces.get()),
                                                 sizeof(float) * 3);
    bool ret = acc_component.accel.Build(mesh->n_faces(), triangle_mesh, triangle_pred);
    assert(ret);

    nanort::BVHBuildStatistics stats = acc_component.accel.GetStatistics();

    ATCG_INFO("BVH statistics:");
    ATCG_INFO("\t# of leaf   nodes: {0}", stats.num_leaf_nodes);
    ATCG_INFO("\t# of branch nodes: {0}", stats.num_branch_nodes);
    ATCG_INFO("\tMax tree depth   : {0}", stats.max_tree_depth);
}

Tracing::HitInfo
Tracing::traceRay(Entity entity, const glm::vec3& ray_origin, const glm::vec3& ray_dir, float t_min, float t_max)
{
    HitInfo result = {};

    if(!entity.hasComponent<AccelerationStructureComponent>())
    {
        ATCG_WARN("Entity does not have an acceleration structure. Aborting...");
        return result;
    }

    auto& acc_component = entity.getComponent<AccelerationStructureComponent>();
    nanort::Ray<float> ray;
    memcpy(ray.org, glm::value_ptr(ray_origin), sizeof(glm::vec3));
    memcpy(ray.dir, glm::value_ptr(ray_dir), sizeof(glm::vec3));

    ray.min_t = t_min;
    ray.max_t = t_max;

    nanort::TriangleIntersector<> triangle_intersector(reinterpret_cast<const float*>(acc_component.vertices.get()),
                                                       reinterpret_cast<const uint32_t*>(acc_component.faces.get()),
                                                       sizeof(float) * 3);
    nanort::TriangleIntersection<> isect;
    bool hit = acc_component.accel.Traverse(ray, triangle_intersector, &isect);

    if(!hit) return result;

    result.hit           = true;
    result.p             = ray_origin + isect.t * ray_dir;
    result.primitive_idx = isect.prim_id;
    return result;
}
}    // namespace atcg