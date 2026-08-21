#include <Math/Tracing.h>
#include <Scene/Components.h>

namespace atcg
{

namespace detail
{
ATCG_INLINE glm::vec3 read_vec3(int index, const torch::Tensor& tensor)
{
    assert(tensor.dim() == 2);
    assert(tensor.size(1) == 3);
    assert(index < tensor.size(0));
    return glm::vec3(tensor[index][0].item<float>(), tensor[index][1].item<float>(), tensor[index][2].item<float>());
}
}    // namespace detail

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
    atcg::ref_ptr<Graph> mesh = geometry_component.graph();
    if(mesh->type() != GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH)
    {
        ATCG_WARN("Can only create BVH for triangles. Aborting...");
        return;
    }

    acc_component.vertices = mesh->getHostPositions().clone();
    acc_component.normals  = mesh->getHostNormals().clone();
    acc_component.uvs      = mesh->getHostUVs().clone();
    acc_component.faces    = mesh->getHostFaces().clone();

    nanort::TriangleMesh<float> triangle_mesh(reinterpret_cast<const float*>(acc_component.vertices.data_ptr()),
                                              reinterpret_cast<const uint32_t*>(acc_component.faces.data_ptr()),
                                              sizeof(float) * 3);
    nanort::TriangleSAHPred<float> triangle_pred(reinterpret_cast<const float*>(acc_component.vertices.data_ptr()),
                                                 reinterpret_cast<const uint32_t*>(acc_component.faces.data_ptr()),
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

    nanort::TriangleIntersector<> triangle_intersector(
        reinterpret_cast<const float*>(acc_component.vertices.data_ptr()),
        reinterpret_cast<const uint32_t*>(acc_component.faces.data_ptr()),
        sizeof(float) * 3);
    nanort::TriangleIntersection<> isect;
    bool hit = acc_component.accel.Traverse(ray, triangle_intersector, &isect);

    if(!hit) return result;

    glm::u32vec3 face = glm::u32vec3(acc_component.faces[isect.prim_id][0].item<float>(),
                                     acc_component.faces[isect.prim_id][1].item<float>(),
                                     acc_component.faces[isect.prim_id][2].item<float>());

    const glm::vec3& n0 = detail::read_vec3(face[0], acc_component.normals);
    const glm::vec3& n1 = detail::read_vec3(face[1], acc_component.normals);
    const glm::vec3& n2 = detail::read_vec3(face[2], acc_component.normals);

    const glm::vec3& uv0 = detail::read_vec3(face[0], acc_component.uvs);
    const glm::vec3& uv1 = detail::read_vec3(face[1], acc_component.uvs);
    const glm::vec3& uv2 = detail::read_vec3(face[2], acc_component.uvs);


    result.position           = ray_origin + isect.t * ray_dir;
    result.incoming_direction = ray_dir;
    result.incoming_distance  = isect.t;
    result.normal             = glm::normalize(n0 * (1.0f - isect.u - isect.v) + n1 * isect.u + n2 * isect.v);
    result.barys              = glm::vec2(isect.u, isect.v);
    result.uv                 = glm::vec2(uv0 * (1.0f - isect.u - isect.v) + uv1 * isect.u + uv2 * isect.v);
    result.primitive_idx      = isect.prim_id;
    return result;
}
}    // namespace atcg