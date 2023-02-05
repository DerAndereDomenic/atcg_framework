#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <queue>

#include <numeric>
#include <random>

class G12Layer : public atcg::Layer
{
public:
    using AssignmentMap = std::vector<std::vector<uint32_t>>;

    G12Layer(const std::string& name) : atcg::Layer(name) {}

    template<typename T>
    struct LaplaceCotan
    {
        T clampCotan(T v)
        {
            const T bound = T(19.1);
            return (v < -bound ? -bound : (v > bound ? bound : v));
        }

        T triangleCotan(const atcg::TriMesh::Point& v0, const atcg::TriMesh::Point& v1, const atcg::TriMesh::Point& v2)
        {
            /// Exercise: Compute the cotan values for a triangle
            ///           Hint: cotan = <edge0,edge1>/Area(Triangle)
            ///           You can use atcg::areaFromMetric<T> to compute the triangle area
            const auto d0   = v0 - v2;
            const auto d1   = v1 - v2;
            const auto d2   = v1 - v0;
            const auto area = atcg::areaFromMetric<T>(d0.norm(), d1.norm(), d2.norm());
            if(area > 1e-5) return clampCotan(d0.dot(d1) / area) / 2.f;
            return T(1e-5);
        }

        atcg::Laplacian<T> calculate(const std::shared_ptr<atcg::Mesh>& mesh)
        {
            std::vector<Eigen::Triplet<T>> edge_weights;

            for(auto e_it = mesh->edges_begin(); e_it != mesh->edges_end(); ++e_it)
            {
                /// Exercise: Compute the edge weights using cotan weights
                ///           Remember to check for boundary edges and handle them accordingly
                uint32_t i = e_it->v0().idx();
                uint32_t j = e_it->v1().idx();

                const auto h0 = e_it->h0();
                const auto h1 = e_it->h1();

                const auto p0 = h0.to();
                const auto p1 = h1.to();

                T weight = 0;
                if(!mesh->is_boundary(h0))
                {
                    const auto p2 = h0.next().to();
                    weight += triangleCotan(mesh->point(p0), mesh->point(p1), mesh->point(p2)) / 2.f;
                }

                if(!mesh->is_boundary(h1))
                {
                    const auto p2 = h1.next().to();
                    weight += triangleCotan(mesh->point(p0), mesh->point(p1), mesh->point(p2)) / 2.f;
                }

                edge_weights.emplace_back(i, j, weight);
                edge_weights.emplace_back(j, i, weight);
                edge_weights.emplace_back(i, i, -weight);
                edge_weights.emplace_back(j, j, -weight);
            }

            size_t N = mesh->n_vertices();

            atcg::Laplacian<T> laplace;
            laplace.S.resize(N, N);
            laplace.M.resize(N, N);

            laplace.S.setFromTriplets(edge_weights.begin(), edge_weights.end());
            /// Exercise: Set M = |diagonal(S)|
            laplace.M = laplace.S.diagonal().cwiseAbs().asDiagonal();

            return laplace;
        }
    };

    AssignmentMap kmeans(const Eigen::MatrixXd& X, const uint32_t k = 6, const uint32_t max_iterations = 50)
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<uint32_t> dist(0, X.rows());

        std::vector<Eigen::VectorXd> centers;

        // Init
        AssignmentMap assignments;
        for(uint32_t i = 0; i < k; ++i)
        {
            uint32_t center_idx = dist(rng);
            centers.push_back(X.col(center_idx));
            assignments.push_back(std::vector<uint32_t>());
        }

        for(uint32_t it = 0; it < max_iterations; ++it)
        {
            std::cout << it << "\n";
            for(uint32_t i = 0; i < k; ++i) assignments[i].clear();

            // Assignment
            for(uint32_t i = 0; i < X.cols(); ++i)
            {
                double MIN_DIST = std::numeric_limits<double>::infinity();
                uint32_t assign = 0;

                for(uint32_t j = 0; j < k; ++j)
                {
                    double dist = (X.col(i) - centers[j]).norm();
                    if(dist < MIN_DIST)
                    {
                        MIN_DIST = dist;
                        assign   = j;
                    }
                }

                assignments[assign].push_back(i);
            }

            // Update
            for(uint32_t i = 0; i < k; ++i)
            {
                Eigen::VectorXd mean(X.rows());
                mean.setZero();

                for(uint32_t j = 0; j < assignments[i].size(); ++j) { mean += X.col(assignments[i][j]); }
                mean       = mean / static_cast<double>(assignments[i].size());
                centers[i] = mean;
            }
        }

        return assignments;
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = std::make_shared<atcg::CameraController>(aspect_ratio);

        mesh = atcg::IO::read_mesh("res/suzanne_blender.obj");
        mesh->request_vertex_colors();

        atcg::Laplacian laplace       = LaplaceCotan<double>().calculate(mesh);
        Eigen::SparseMatrix<double> L = laplace.M.cwiseInverse() * laplace.S;

        /*Eigen::EigenSolver<Eigen::MatrixXd> solver;
        solver.compute(L.toDense());

        Eigen::MatrixXd V = solver.eigenvectors().real();
        Eigen::VectorXd w = solver.eigenvalues().real();*/

        Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeFullV> solver(L);

        Eigen::VectorXd w = solver.singularValues();
        Eigen::MatrixXd V = solver.matrixV();

        // Eigen::MatrixXd GPS =
        // (w+Eigen::VectorXd::Constant(w.size(),1e-12)).cwiseAbs().cwiseSqrt().cwiseInverse().asDiagonal() *
        // V.transpose();
        Eigen::MatrixXd GPS =
            (w + Eigen::VectorXd::Constant(w.size(), 1e-12)).cwiseInverse().asDiagonal() * V.transpose();

        AssignmentMap assignments = kmeans(GPS);

        atcg::Mesh::Color colors[6] = {atcg::Mesh::Color {255, 0, 0},
                                       atcg::Mesh::Color {0, 255, 0},
                                       atcg::Mesh::Color {0, 0, 255},
                                       atcg::Mesh::Color {255, 0, 255},
                                       atcg::Mesh::Color {0, 255, 255},
                                       atcg::Mesh::Color {255, 255, 0}};

        for(uint32_t i = 0; i < 6; ++i)
        {
            for(uint32_t j = 0; j < assignments[i].size(); ++j)
            {
                mesh->set_color(atcg::Mesh::VertexHandle(assignments[i][j]), colors[i]);
            }
        }

        mesh->uploadData();
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        if(mesh && render_faces)
            atcg::Renderer::draw(mesh, atcg::ShaderManager::getShader("base"), camera_controller->getCamera());

        if(mesh && render_points)
            atcg::Renderer::drawPoints(mesh,
                                       glm::vec3(0),
                                       atcg::ShaderManager::getShader("base"),
                                       camera_controller->getCamera());

        if(mesh && render_edges) atcg::Renderer::drawLines(mesh, glm::vec3(1), camera_controller->getCamera());
    }

    virtual void onImGuiRender() override
    {
        ImGui::BeginMainMenuBar();

        if(ImGui::BeginMenu("Rendering"))
        {
            ImGui::MenuItem("Show Render Settings", nullptr, &show_render_settings);

            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();

        if(show_render_settings)
        {
            ImGui::Begin("Settings", &show_render_settings);

            ImGui::Checkbox("Render Vertices", &render_points);
            ImGui::Checkbox("Render Edges", &render_edges);
            ImGui::Checkbox("Render Mesh", &render_faces);
            ImGui::End();
        }
    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event& event) override { camera_controller->onEvent(event); }

private:
    std::shared_ptr<atcg::CameraController> camera_controller;
    std::shared_ptr<atcg::Mesh> mesh;

    bool show_render_settings = true;
    bool render_faces         = true;
    bool render_points        = false;
    bool render_edges         = false;
};

class G12 : public atcg::Application
{
public:
    G12() : atcg::Application() { pushLayer(new G12Layer("Layer")); }

    ~G12() {}
};

atcg::Application* atcg::createApplication()
{
    return new G12;
}