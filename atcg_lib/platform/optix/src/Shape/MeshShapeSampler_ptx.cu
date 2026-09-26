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

    glm::u32vec3* faces_3d   = sbt_data->faces_3d;
    glm::u32vec3* faces_uv   = sbt_data->faces_uv;
    glm::vec3* positions     = sbt_data->positions;
    glm::vec3* UVs           = sbt_data->uvs;
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
    glm::u32vec3 vertex_indices = faces_3d[triangle_index];
    glm::u32vec3 uv_indices     = faces_uv[triangle_index];

    // Vertex positions of selected triangle
    glm::vec3 P0 = positions[vertex_indices.x];
    glm::vec3 P1 = positions[vertex_indices.y];
    glm::vec3 P2 = positions[vertex_indices.z];

    // Vertex UVs of selected triangle
    glm::vec3 UV0 = UVs[uv_indices.x];
    glm::vec3 UV1 = UVs[uv_indices.y];
    glm::vec3 UV2 = UVs[uv_indices.z];

    glm::vec3 uvs =
        (1.0f - triangle_barys.x - triangle_barys.y) * UV0 + triangle_barys.x * UV1 + triangle_barys.y * UV2;

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
    result.pdf_dA   = 1.0f / sbt_data->total_area;
    result.uvs      = uvs;

    return result;
}

extern "C" __device__ atcg::EdgeSampleResult __direct_callable__sample_edge_mesh(atcg::PCG32& rng)
{
    const atcg::MeshSamplerData* sbt_data = *reinterpret_cast<const atcg::MeshSamplerData**>(optixGetSbtDataPointer());

    atcg::EdgeSampleResult result;

    float* edge_cdf    = sbt_data->edge_cdf;
    uint32_t num_edges = sbt_data->num_edges;

    glm::u32vec2* edges      = sbt_data->edges;
    glm::vec3* positions     = sbt_data->positions;
    glm::mat4 local_to_world = sbt_data->local_to_world;

    // Select the edge to sample a direction from uniformly at random, proportional to its length
    uint32_t edge_index = 0;
    // Sample the barycentric coordinates on the edge uniformly.
    float edge_barys = rng.next1d();

    edge_index = atcg::Math::binary_search(edge_cdf, rng.next1d(), num_edges);

    // Compute the `light_position` using the triangle_index and the triangle_barys on the mesh:

    // Indices of triangle vertices in the mesh
    glm::u32vec2 vertex_indices = edges[edge_index];

    // Vertex positions of selected triangle
    glm::vec3 P0 = positions[vertex_indices.x];
    glm::vec3 P1 = positions[vertex_indices.y];

    // Compute local position
    glm::vec3 local_position = (1.0f - edge_barys) * P0 + edge_barys * P1;

    glm::vec3 local_tangent = glm::normalize(P1 - P0);

    // Transform local position to world position
    glm::vec3 position = glm::vec3(local_to_world * glm::vec4(local_position, 1));

    // Transform local tangent to world tangent
    glm::vec3 tangent = glm::normalize(glm::vec3(local_to_world * glm::vec4(local_tangent, 0)));

    glm::i32vec2 face_indices = sbt_data->edge_faces[edge_index];

    if(face_indices.x != -1)
    {
        result.valid_left = true;

        glm::u32vec3 face_vertex_indices = sbt_data->faces_3d[face_indices.x];

        glm::vec3 p0 = positions[face_vertex_indices.x];
        glm::vec3 p1 = positions[face_vertex_indices.y];
        glm::vec3 p2 = positions[face_vertex_indices.z];

        glm::vec3 local_normal = glm::cross(p1 - p0, p2 - p0);
        result.normal_left     = glm::normalize(glm::transpose(glm::mat3(sbt_data->world_to_local)) * local_normal);
    }

    if(face_indices.y != -1)
    {
        result.valid_right = true;

        glm::u32vec3 face_vertex_indices = sbt_data->faces_3d[face_indices.y];
        glm::vec3 p0                     = positions[face_vertex_indices.x];
        glm::vec3 p1                     = positions[face_vertex_indices.y];
        glm::vec3 p2                     = positions[face_vertex_indices.z];

        glm::vec3 local_normal = glm::cross(p1 - p0, p2 - p0);
        result.normal_right    = glm::normalize(glm::transpose(glm::mat3(sbt_data->world_to_local)) * local_normal);
    }

    // Assemble sampling result
    result.position = position;
    result.tangent  = tangent;
    result.pdf_dl   = 1.0f / sbt_data->total_edge_length;

    return result;
}

extern "C" __device__ float __direct_callable__evalpdf_point_mesh(const glm::vec3& position)
{
    const atcg::MeshSamplerData* sbt_data = *reinterpret_cast<const atcg::MeshSamplerData**>(optixGetSbtDataPointer());

    return 1.0f / sbt_data->total_area;
}

extern "C" __device__ float __direct_callable__evalpdf_edge_mesh(const glm::vec3& position)
{
    const atcg::MeshSamplerData* sbt_data = *reinterpret_cast<const atcg::MeshSamplerData**>(optixGetSbtDataPointer());

    return 1.0f / sbt_data->total_edge_length;
}
