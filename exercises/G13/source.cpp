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

class G13Layer : public atcg::Layer
{
public:

    using AssignmentMap = std::vector<std::vector<uint32_t>>;

    G13Layer(const std::string& name) : atcg::Layer(name) {}

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
            const auto d0 = v0 - v2;
            const auto d1 = v1 - v2;
            const auto d2 = v1 - v0;
            const auto area = atcg::areaFromMetric<T>(d0.norm(), d1.norm(), d2.norm());
            if(area > 1e-5)
                return clampCotan(d0.dot(d1) / area)/2.f;
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

    std::vector<float> linspace(float a, float b, uint32_t steps)
    {
        float step_size = (b-a) / (steps - 1);

        std::vector<float> space(steps);

        for(uint32_t i = 0; i < steps; ++i)
        {
            space[i] = (a + i * step_size);
        }

        return space;
    }

    std::shared_ptr<atcg::Mesh> triangulate(const std::vector<atcg::Mesh::Point>& points)
    {
        std::shared_ptr<atcg::Mesh> mesh = std::make_shared<atcg::Mesh>();
        
        std::vector<atcg::Mesh::VertexHandle> v_handles(points.size());

        for(uint32_t i = 0; i < points.size(); ++i)
            v_handles[i] = mesh->add_vertex({points[i][0], points[i][2], points[i][1]});

        uint32_t grid_size = static_cast<uint32_t>(std::sqrt(points.size())) - 1;

        for(uint32_t grid_x = 0; grid_x < grid_size; ++grid_x)
        {
            for(uint32_t grid_y = 0; grid_y < grid_size; ++grid_y)
            {
                auto v00 = v_handles[grid_x + (grid_size+1) * grid_y];
                auto v10 = v_handles[grid_x + 1 + (grid_size+1) * grid_y];
                auto v01 = v_handles[grid_x + (grid_size+1) * (grid_y + 1)];
                auto v11 = v_handles[grid_x + 1 + (grid_size+1) * (grid_y + 1)];

                mesh->add_face(v00, v01, v10);
                mesh->add_face(v10, v01, v11);
            }
        }

        return mesh;
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        std::vector<float> U = linspace(-1,1,150);
        std::vector<atcg::Mesh::Point> grid;

        for(float u : U)
        {
            for(float v : U)
            {
                grid.push_back({v, u, 0.f});
            }
        }

        mesh = triangulate(grid);

        atcg::Laplacian<double> laplace = LaplaceCotan<double>().calculate(mesh);
        Eigen::SparseMatrix<double> L = laplace.M.cwiseInverse() * laplace.S;
        Eigen::SparseMatrix<double> L2 = L*L;

        double ks = 1.0;
        double kb = 1.0;
        Eigen::SparseMatrix<double> op = -ks * L + kb * L2;

        double edit_radius = 0.3;
        double region_radius = 0.8;

        Eigen::MatrixXd starting_displacement(mesh->n_vertices(), 3);

        Eigen::VectorXd ones = Eigen::VectorXd::Ones(mesh->n_vertices());
        Eigen::VectorXd zeros = Eigen::VectorXd::Zero(mesh->n_vertices());

        for(auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end(); ++v_it)
        {
            atcg::Mesh::Point p = mesh->point(*v_it);
            double distance = p.norm();

            if(distance < edit_radius)
            {
                starting_displacement.row(v_it->idx()) = Eigen::Vector3d(0.0, 1.0, 0.0);
                ones(v_it->idx()) = 0;
                zeros(v_it->idx()) = 1;
            }
            else if(distance > region_radius)
            {
                ones(v_it->idx()) = 0;
                zeros(v_it->idx()) = 1;
            }
        }

        Eigen::SparseMatrix<double> Id(mesh->n_vertices(), mesh->n_vertices());
        Id.setIdentity();
        op = ones.asDiagonal() * op;
        op = op + Id*zeros.asDiagonal();

        Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
        solver.compute(op);
        Eigen::MatrixXd displacement = solver.solve(starting_displacement);

        for(auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end(); ++v_it)
        {
            atcg::Mesh::Point p = mesh->point(*v_it);
            Eigen::Vector3d d = displacement.row(v_it->idx());
            mesh->set_point(*v_it, {p[0] + d(0), p[1] + d(1), p[2] + d(2)});
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
            atcg::Renderer::drawPoints(mesh, glm::vec3(0), atcg::ShaderManager::getShader("base"), camera_controller->getCamera());

        if(mesh && render_edges)
            atcg::Renderer::drawLines(mesh, glm::vec3(1), camera_controller->getCamera());
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
    virtual void onEvent(atcg::Event& event) override
    {
        camera_controller->onEvent(event);
    }

private:
    std::shared_ptr<atcg::CameraController> camera_controller;
    std::shared_ptr<atcg::Mesh> mesh;

    bool show_render_settings = true;
    bool render_faces = true;
    bool render_points = false;
    bool render_edges = false;
};

class G13 : public atcg::Application
{
    public:

    G13()
        :atcg::Application()
    {
        pushLayer(new G13Layer("Layer"));
    }

    ~G13() {}

};

atcg::Application* atcg::createApplication()
{
    return new G13;
}