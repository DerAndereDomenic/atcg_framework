#include "Radiosity.h"

#include <glm/ext/scalar_constants.hpp>

#include <nanort.h>

using namespace atcg;
using namespace torch::indexing;

atcg::ref_ptr<TriMesh> solve_radiosity(const atcg::ref_ptr<TriMesh>& mesh, const torch::Tensor& emission)
{
    const glm::vec3* vertices = mesh->points();
    std::vector<uint32_t> faces;
    for(auto ft: mesh->faces())
    {
        for(auto vt: ft.vertices())
        {
            faces.push_back(vt.idx());
        }
    }

    nanort::TriangleMesh<float> triangle_mesh(reinterpret_cast<const float*>(vertices),
                                              faces.data(),
                                              sizeof(float) * 3);
    nanort::TriangleSAHPred<float> triangle_pred(reinterpret_cast<const float*>(vertices),
                                                 faces.data(),
                                                 sizeof(float) * 3);
    nanort::BVHAccel<float> accel;
    bool ret = accel.Build(mesh->n_faces(), triangle_mesh, triangle_pred);
    assert(ret);

    nanort::BVHBuildStatistics stats = accel.GetStatistics();

    printf("  BVH statistics:\n");
    printf("    # of leaf   nodes: %d\n", stats.num_leaf_nodes);
    printf("    # of branch nodes: %d\n", stats.num_branch_nodes);
    printf("  Max tree depth     : %d\n", stats.max_tree_depth);

    torch::Tensor albedos      = torch::zeros({(int)mesh->n_faces(), 3});
    torch::Tensor form_factors = torch::zeros({(int)mesh->n_faces(), (int)mesh->n_faces()});

    for(auto ft: mesh->faces())
    {
        glm::vec3 color = glm::vec3(0);
        for(auto vt: ft.vertices())
        {
            color += mesh->color(vt) / 255.0f;
        }
        color /= 3.0f;
        albedos.index_put_({ft.idx(), Slice()}, torch::tensor({color.x, color.y, color.z}));
    }

// Compute Form factor matrix
#pragma omp parallel for
    for(int32_t i = 0; i < mesh->n_faces(); ++i)
    {
        for(uint32_t j = i + 1; j < mesh->n_faces(); ++j)
        {
            TriMesh::FaceHandle f1(i);
            TriMesh::FaceHandle f2(j);

            glm::vec3 centroid_f1   = mesh->calc_face_centroid(f1);
            glm::vec3 centroid_f2   = mesh->calc_face_centroid(f2);
            glm::vec3 ray_direction = centroid_f2 - centroid_f1;
            float distance          = glm::length(ray_direction);
            ray_direction /= distance;

            glm::vec3 n1 = mesh->calc_face_normal(f1);
            glm::vec3 n2 = mesh->calc_face_normal(f2);

            float cos_theta_1 = glm::max(0.0f, glm::dot(n1, ray_direction));
            float cos_theta_2 = glm::max(0.0f, -glm::dot(n2, ray_direction));

            // Calculate visiblity
            nanort::Ray<float> ray;
            memcpy(ray.org, reinterpret_cast<float*>(&centroid_f1), sizeof(float) * 3);
            memcpy(ray.dir, reinterpret_cast<float*>(&ray_direction), sizeof(float) * 3);

            ray.min_t = 1e-4f;
            ray.max_t = distance - 1e-4f;

            nanort::TriangleIntersector<> triangle_intersector(reinterpret_cast<const float*>(vertices),
                                                               faces.data(),
                                                               sizeof(float) * 3);
            nanort::TriangleIntersection<> isect;
            bool hit = accel.Traverse(ray, triangle_intersector, &isect);

            float G = 0.0f;
            if(!hit && cos_theta_1 > 0 && cos_theta_2 > 0)
            {
                G = cos_theta_1 * cos_theta_2 / (distance * distance * glm::pi<float>());
            }

            form_factors.index_put_({(int)i, (int)j}, G * mesh->calc_face_area(f2));
            form_factors.index_put_({(int)j, (int)i}, G * mesh->calc_face_area(f1));
        }
    }

    ATCG_TRACE("Solving");

    torch::Tensor solution = torch::zeros({(int)mesh->n_faces(), 3});
    torch::Tensor FE       = torch::zeros({(int)mesh->n_faces(), 3});
    torch::Tensor update   = torch::zeros({(int)mesh->n_faces(), 3});

    for(int iter = 0; iter < 50; ++iter)
    {
        ATCG_TRACE("{0}", iter);

        FE = torch::matmul(form_factors, solution);

        FE                   = albedos * FE;
        torch::Tensor update = (emission - solution + FE);

        if(torch::norm(update).item<float>() < 1e-4f) break;

        solution += update;
    }

    printf("Done\n");

    // Copy mesh in a super convuluted way because the documentation doesn't state how to do it right
    atcg::ref_ptr<TriMesh> result = atcg::make_ref<TriMesh>();
    result->operator=(*mesh);
    result->request_face_colors();

    for(auto ft = result->faces_begin(); ft != result->faces_end(); ++ft)
    {
        torch::Tensor radiostiy = solution.index({ft->idx(), Slice()});
        glm::vec3 color         = glm::vec3(radiostiy.index({0}).item<float>(),
                                    radiostiy.index({1}).item<float>(),
                                    radiostiy.index({2}).item<float>());
        color                   = glm::pow(1.0f - glm::exp(-color), glm::vec3(1.0f / 2.4f));
        for(auto vt = ft->vertices().begin(); vt != ft->vertices().end(); ++vt)
            result->set_color(*vt, 255.0f * color);
    }

    return result;
}