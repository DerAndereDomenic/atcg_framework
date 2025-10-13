#include "Radiosity.h"

#include <glm/ext/scalar_constants.hpp>

#ifdef ATCG_ENABLE_OPTIX
    #include "RadiosityRayGenerator.h"
#endif

#include <nanort.h>

using namespace atcg;
using namespace torch::indexing;

atcg::ref_ptr<atcg::TriMesh> solve_radiosity(const atcg::ref_ptr<atcg::TriMesh>& mesh, const torch::Tensor& emission)
{
#ifdef ATCG_ENABLE_OPTIX
    return solve_radiosity_gpu(mesh, emission);
#else
    return solve_radiosity_cpu(mesh, emission);
#endif
}

atcg::ref_ptr<atcg::TriMesh> solve_radiosity_gpu(const atcg::ref_ptr<atcg::TriMesh>& mesh,
                                                 const torch::Tensor& emission)
{
    atcg::ref_ptr<TriMesh> result = atcg::make_ref<TriMesh>();
#ifdef ATCG_ENABLE_OPTIX
    auto optx_context = atcg::RaytracingContextManager::createContext();
    auto pipeline     = atcg::make_ref<atcg::RayTracingPipeline>(optx_context);
    auto sbt          = atcg::make_ref<atcg::ShaderBindingTable>();

    RadiosityRayGenerator generator(optx_context);
    generator.setMesh(mesh);
    generator.initializePipeline(pipeline, sbt);

    pipeline->createPipeline();
    sbt->createSBT();

    torch::Tensor form_factors =
        torch::zeros({(int)mesh->n_faces(), (int)mesh->n_faces()}, atcg::TensorOptions::floatDeviceOptions());
    atcg::Dictionary dict;
    dict.setValue("output", form_factors);
    generator.generateRays(dict);

    ATCG_TRACE("Solving");

    torch::Tensor albedos = torch::zeros({(int)mesh->n_faces(), 3});
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
    albedos = albedos.to(atcg::GPU);

    torch::Tensor solution = torch::zeros({(int)mesh->n_faces(), 3}, atcg::TensorOptions::floatDeviceOptions());
    torch::Tensor E        = emission.to(atcg::GPU);

    for(int iter = 0; iter < 50; ++iter)
    {
        ATCG_TRACE("{0}", iter);

        auto FE = torch::matmul(form_factors, solution);

        FE                   = albedos * FE;
        torch::Tensor update = (E - solution + FE);

        if(torch::norm(update).item<float>() < 1e-4f) break;

        solution += update;
    }
    solution = solution.to(atcg::CPU);
    printf("Done\n");

    // Copy mesh in a super convuluted way because the documentation doesn't state how to do it right
    result->operator=(*mesh);
    result->request_vertex_colors();

    for(auto vt = result->vertices_begin(); vt != result->vertices_end(); ++vt)
    {
        glm::vec3 color(0.0f);
        uint32_t n_faces = 0;
        for(auto ft = vt->faces().begin(); ft != vt->faces().end(); ++ft)
        {
            torch::Tensor radiostiy = solution.index({ft->idx(), Slice()});
            color += glm::vec3(radiostiy[0].item<float>(), radiostiy[1].item<float>(), radiostiy[2].item<float>());
            ++n_faces;
        }

        color /= float(n_faces);
        color = glm::pow(1.0f - glm::exp(-color), glm::vec3(1.0f / 2.4f));
        result->set_color(*vt, color * 255.0f);
    }
#endif
    return result;
}

atcg::ref_ptr<TriMesh> solve_radiosity_cpu(const atcg::ref_ptr<TriMesh>& mesh, const torch::Tensor& emission)
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
    atcg::WorkerPool pool(64);
    pool.start();

    for(int32_t i = 0; i < mesh->n_faces(); ++i)
    {
        for(uint32_t j = i + 1; j < mesh->n_faces(); ++j)
        {
            pool.pushJob(
                [i, j, mesh, &vertices, &faces, &accel, &form_factors]()
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
                });
        }
    }

    pool.waitDone();
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
    result->request_vertex_colors();

    for(auto vt = result->vertices_begin(); vt != result->vertices_end(); ++vt)
    {
        glm::vec3 color(0.0f);
        uint32_t n_faces = 0;
        for(auto ft = vt->faces().begin(); ft != vt->faces().end(); ++ft)
        {
            torch::Tensor radiostiy = solution.index({ft->idx(), Slice()});
            color += glm::vec3(radiostiy[0].item<float>(), radiostiy[1].item<float>(), radiostiy[2].item<float>());
            ++n_faces;
        }

        color /= float(n_faces);
        color = glm::pow(1.0f - glm::exp(-color), glm::vec3(1.0f / 2.4f));
        result->set_color(*vt, color * 255.0f);
    }

    return result;
}