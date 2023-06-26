#include "Radiosity.h"

#include <glm/ext/scalar_constants.hpp>

#include <nanort.h>

// using namespace atcg;

// atcg::ref_ptr<Mesh> solve_radiosity(const atcg::ref_ptr<Mesh>& mesh, const Eigen::MatrixX3f& emission)
// {
//     const glm::vec3* vertices = mesh->points();
//     std::vector<uint32_t> faces;
//     for(auto ft: mesh->faces())
//     {
//         for(auto vt: ft.vertices()) { faces.push_back(vt.idx()); }
//     }

//     nanort::TriangleMesh<float> triangle_mesh(reinterpret_cast<const float*>(vertices),
//                                               faces.data(),
//                                               sizeof(float) * 3);
//     nanort::TriangleSAHPred<float> triangle_pred(reinterpret_cast<const float*>(vertices),
//                                                  faces.data(),
//                                                  sizeof(float) * 3);
//     nanort::BVHAccel<float> accel;
//     bool ret = accel.Build(mesh->n_faces(), triangle_mesh, triangle_pred);
//     assert(ret);

//     nanort::BVHBuildStatistics stats = accel.GetStatistics();

//     printf("  BVH statistics:\n");
//     printf("    # of leaf   nodes: %d\n", stats.num_leaf_nodes);
//     printf("    # of branch nodes: %d\n", stats.num_branch_nodes);
//     printf("  Max tree depth     : %d\n", stats.max_tree_depth);

//     Eigen::MatrixX3f albedos(mesh->n_faces(), 3);
//     Eigen::MatrixXf form_factors = Eigen::MatrixXf::Zero(mesh->n_faces(), mesh->n_faces());

//     for(auto ft: mesh->faces())
//     {
//         glm::vec3 color = glm::vec3(0);
//         for(auto vt: ft.vertices()) { color += mesh->color(vt) / 255.0f; }
//         color /= 3.0f;
//         albedos.row(ft.idx()) = Eigen::Vector3f {color.x, color.y, color.z};
//     }

// // Compute Form factor matrix
// #pragma omp parallel for
//     for(int32_t i = 0; i < mesh->n_faces(); ++i)
//     {
//         for(uint32_t j = i + 1; j < mesh->n_faces(); ++j)
//         {
//             Mesh::FaceHandle f1(i);
//             Mesh::FaceHandle f2(j);

//             glm::vec3 centroid_f1   = mesh->calc_face_centroid(f1);
//             glm::vec3 centroid_f2   = mesh->calc_face_centroid(f2);
//             glm::vec3 ray_direction = centroid_f2 - centroid_f1;
//             float distance          = glm::length(ray_direction);
//             ray_direction /= distance;

//             glm::vec3 n1 = mesh->calc_face_normal(f1);
//             glm::vec3 n2 = mesh->calc_face_normal(f2);

//             float cos_theta_1 = glm::max(0.0f, glm::dot(n1, ray_direction));
//             float cos_theta_2 = glm::max(0.0f, -glm::dot(n2, ray_direction));

//             // Calculate visiblity
//             nanort::Ray<float> ray;
//             memcpy(ray.org, reinterpret_cast<float*>(&centroid_f1), sizeof(float) * 3);
//             memcpy(ray.dir, reinterpret_cast<float*>(&ray_direction), sizeof(float) * 3);

//             ray.min_t = 1e-4f;
//             ray.max_t = distance - 1e-4f;

//             nanort::TriangleIntersector<> triangle_intersector(reinterpret_cast<const float*>(vertices),
//                                                                faces.data(),
//                                                                sizeof(float) * 3);
//             nanort::TriangleIntersection<> isect;
//             bool hit = accel.Traverse(ray, triangle_intersector, &isect);

//             float G = 0.0f;
//             if(!hit && cos_theta_1 > 0 && cos_theta_2 > 0)
//             {
//                 G = cos_theta_1 * cos_theta_2 / (distance * distance * glm::pi<float>());
//             }

//             form_factors(i, j) = G * mesh->calc_face_area(f2);
//             form_factors(j, i) = G * mesh->calc_face_area(f1);
//         }
//     }

//     ATCG_TRACE("Solving");

//     Eigen::MatrixX3f solution = Eigen::MatrixX3f::Zero(mesh->n_faces(), 3);
//     Eigen::MatrixX3f FE       = Eigen::MatrixX3f::Zero(mesh->n_faces(), 3);
//     Eigen::MatrixX3f update   = Eigen::MatrixX3f::Zero(mesh->n_faces(), 3);

//     for(int iter = 0; iter < 50; ++iter)
//     {
//         ATCG_TRACE("{0}", iter);

//         FE = form_factors * solution;

//         FE                      = albedos.array() * FE.array();
//         Eigen::MatrixX3f update = (emission - solution + FE);

//         if(update.norm() < 1e-4f) break;

//         solution += update;
//     }

//     printf("Done\n");

//     // Copy mesh in a super convuluted way because the documentation doesn't state how to do it right
//     atcg::ref_ptr<Mesh> result = atcg::make_ref<Mesh>();
//     result->operator=(**mesh);
//     result->request_face_colors();

//     for(auto ft = result->faces_begin(); ft != result->faces_end(); ++ft)
//     {
//         Eigen::Vector3f radiostiy = solution.row(ft->idx());
//         glm::vec3 color           = glm::vec3(radiostiy.x(), radiostiy.y(), radiostiy.z());
//         color                     = glm::pow(1.0f - glm::exp(-color), glm::vec3(1.0f / 2.4f));
//         for(auto vt = ft->vertices().begin(); vt != ft->vertices().end(); ++vt) result->set_color(*vt, 255.0f *
//         color);
//     }

//     result->uploadData();
//     return result;
// }