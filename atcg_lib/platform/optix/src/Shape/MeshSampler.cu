#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>
#include <Utils/HostDevice.h>
#include <Core/SurfaceInteraction.h>

#include <Shape/ShapeSamplerVPtrTable.cuh>
#include <Shape/MeshSamplerData.cuh>
#include <DataStructure/Frame.h>
#include <BSDF/Sampling.h>


extern "C" __device__ atcg::ShapeSampleResult __direct_callable__sample_point_mesh(atcg::PCG32& rng)
{
    const atcg::MeshSamplerData* sbt_data = *reinterpret_cast<const atcg::MeshSamplerData**>(optixGetSbtDataPointer());

    atcg::ShapeSampleResult result;

    float* mesh_cdf    = sbt_data->mesh_cdf;
    uint32_t num_faces = sbt_data->num_faces;

    glm::u32vec3* faces      = sbt_data->faces;
    glm::vec3* positions     = sbt_data->positions;
    glm::mat4 local_to_world = sbt_data->local_to_world;
    glm::mat4 world_to_local = sbt_data->world_to_local;

    // Select the triangle to sample a direction from uniformly at random, proportional to its surface area
    uint32_t triangle_index = 0;
    // Sample the barycentric coordinates on the triangle uniformly.
    glm::vec2 triangle_barys = glm::vec2(0, 0);

    triangle_index = atcg::Math::binary_search(mesh_cdf, rng.next1d(), num_faces);

    triangle_barys = rng.next2d();
    // Mirror barys at diagonal line to cover a triangle instead of a square
    if(triangle_barys.x + triangle_barys.y > 1) triangle_barys = glm::vec2(1) - triangle_barys;


    // Compute the `light_position` using the triangle_index and the triangle_barys on the mesh:

    // Indices of triangle vertices in the mesh
    glm::u32vec3 vertex_indices = faces[triangle_index];

    // Vertex positions of selected triangle
    glm::vec3 P0 = positions[vertex_indices.x];
    glm::vec3 P1 = positions[vertex_indices.y];
    glm::vec3 P2 = positions[vertex_indices.z];

    // Compute local position
    glm::vec3 local_position =
        (1.0f - triangle_barys.x - triangle_barys.y) * P0 + triangle_barys.x * P1 + triangle_barys.y * P2;
    // Transform local position to world position
    glm::vec3 position = glm::vec3(local_to_world * glm::vec4(local_position, 1));

    // Compute local normal
    glm::vec3 local_normal = glm::cross(P1 - P0, P2 - P0);
    // Normals are transformed by (A^-1)^T instead of A
    glm::vec3 normal = glm::normalize(glm::transpose(glm::mat3(world_to_local)) * local_normal);

    // Assemble sampling result
    result.position = position;
    result.normal   = normal;
    result.pdf_A    = 1.0f / sbt_data->total_area;

    return result;
}

extern "C" __device__ atcg::EdgeSampleResult __direct_callable__sample_edge_mesh(atcg::PCG32& rng)
{
    const atcg::MeshSamplerData* sbt_data = *reinterpret_cast<const atcg::MeshSamplerData**>(optixGetSbtDataPointer());

    // TODO

    return atcg::EdgeSampleResult();
}
