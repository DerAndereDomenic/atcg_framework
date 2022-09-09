#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <queue>

#include <numeric>

using VertexHandle = atcg::Mesh::VertexHandle;
using EdgeHandle = atcg::Mesh::EdgeHandle;

template<typename T>
struct LaplaceCotan
{
    T clampCotan(T v)
    {
        const T bound = 19.1;
        return (v < -bound ? -bound : (v > bound ? bound : v));
    }

    T areaFromMetric(T a, T b, T c) {
        //Numerically stable herons formula for area of triangle with side lengths a, b and c.
        if (a < b) std::swap(a, b);
        if (a < c) std::swap(a, c);
        if (b < c) std::swap(b, c);

        double p = std::sqrt(std::abs((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c)))) / 4;
        return p;
    }

    T triangleCotan(const atcg::TriMesh::Point& v0, const atcg::TriMesh::Point& v1, const atcg::TriMesh::Point& v2)
    {
        const auto d0 = v0 - v2;
        const auto d1 = v1 - v2;
        const auto d2 = v1 - v0;
        const auto area = areaFromMetric(d0.norm(), d1.norm(), d2.norm());
        if(area > 1e-5)
            return clampCotan(d0.dot(d1) / area);
        return 1e-5;
    }

    atcg::Laplacian<T> calculate(const std::shared_ptr<atcg::Mesh>& mesh)
    {
        std::vector<Eigen::Triplet<T>> edge_weights;

        for(auto e_it = mesh->edges_begin(); e_it != mesh->edges_end(); ++e_it)
        {
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
                weight += triangleCotan(mesh->point(p0), mesh->point(p1), mesh->point(p2));
            }

            if(!mesh->is_boundary(h1))
            {
                const auto p2 = h1.next().to();
                weight += triangleCotan(mesh->point(p0), mesh->point(p1), mesh->point(p2));
            }

            edge_weights.emplace_back(i, j, weight);
            edge_weights.emplace_back(j, i, weight);
            edge_weights.emplace_back(i, i, -weight);
            edge_weights.emplace_back(j, j, -weight);
        }

        std::vector<Eigen::Triplet<T>> vertex_weights;

        for(auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end(); ++v_it)
        {
            T weight = 0;

            for(const auto& e : v_it->edges())
            {
                const auto h0 = e.h0();
                const auto h1 = e.h1();

                const auto p0 = h0.to();
                const auto p1 = h1.to();

                if(!mesh->is_boundary(h0))
                {
                    const auto p2 = h0.next().to();
                    weight += triangleCotan(mesh->point(p0), mesh->point(p1), mesh->point(p2));
                }

                if(!mesh->is_boundary(h1))
                {
                    const auto p2 = h1.next().to();
                    weight += triangleCotan(mesh->point(p0), mesh->point(p1), mesh->point(p2));
                }
            }
            vertex_weights.emplace_back(v_it->idx(), v_it->idx(), weight);
        }

        size_t N = mesh->n_vertices();

        atcg::Laplacian<T> laplace;
        laplace.S.resize(N, N);
        laplace.M.resize(N, N);

        laplace.S.setFromTriplets(edge_weights.begin(), edge_weights.end());
        laplace.M.setFromTriplets(vertex_weights.begin(), vertex_weights.end());

        return laplace;
    }
};

class G06Layer : public atcg::Layer
{
public:

    G06Layer(const std::string& name) : atcg::Layer(name) {}

    void color_mesh(const std::shared_ptr<atcg::Mesh>& p_mesh, const Eigen::VectorXf& u)
    {
        float max_abs_value = std::max(u.maxCoeff(), -u.minCoeff());
        for(auto vh : p_mesh->vertices())
        {
            if(u[vh.idx()] > 0)
                p_mesh->set_color(vh, { u[vh.idx()] / max_abs_value * 255, 0, 0});
            else
                p_mesh->set_color(vh, { 0, 0, - u[vh.idx()] / max_abs_value * 255});
        }
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        mesh = std::make_shared<atcg::Mesh>();
        OpenMesh::IO::read_mesh(*mesh.get(), "res/plane.obj");
        mesh->request_vertex_colors();

        mesh_explicit_large = std::make_shared<atcg::Mesh>();
        OpenMesh::IO::read_mesh(*mesh_explicit_large.get(), "res/plane.obj");
        mesh_explicit_large->request_vertex_colors();

        mesh_explicit_small = std::make_shared<atcg::Mesh>();
        OpenMesh::IO::read_mesh(*mesh_explicit_small.get(), "res/plane.obj");
        mesh_explicit_small->request_vertex_colors();

        mesh_implicit_large = std::make_shared<atcg::Mesh>();
        OpenMesh::IO::read_mesh(*mesh_implicit_large.get(), "res/plane.obj");
        mesh_implicit_large->request_vertex_colors();

        int start_idx = 0;
        float min_dist = std::numeric_limits<float>::infinity();
        for(auto vh : mesh->vertices())
        {
            float dist = mesh->point(vh).norm();
            if(dist < min_dist)
            {
                start_idx = vh.idx();
                min_dist = dist;
            }
        }

        int end_idx = 0;
        float max_dist = -std::numeric_limits<float>::infinity();
        for(auto vh : mesh->vertices())
        {
            float dist = mesh->point(vh).norm();
            if(dist > max_dist)
            {
                end_idx = vh.idx();
                max_dist = dist;
            }
        }

        laplacian = LaplaceCotan<float>().calculate(mesh);

        Eigen::SparseMatrix<float> L = laplacian.M.cwiseInverse() * laplacian.S;

        Eigen::VectorXf u0(mesh->n_vertices());
        u0.setZero();
        u0[start_idx] = 1.0f;
        u0[end_idx] = -1.0f;

        float cfl_timestep = std::numeric_limits<float>::infinity();
        for(auto e_it = mesh->edges_begin(); e_it != mesh->edges_end(); ++e_it)
        {
            float length = mesh->calc_edge_length(*e_it);
            if(length < cfl_timestep)
            {
                cfl_timestep = length;
            }
        }

        float delta_small = 0.9f * cfl_timestep;
        float delta_large = 10.0f * cfl_timestep;
        float time = 500.0f;

        int steps_small = time/delta_small;
        int steps_large = time/delta_large;

        Eigen::VectorXf u_explicit_small = u0;
        Eigen::VectorXf u_explicit_large = u0;
        Eigen::VectorXf u_implict_large = u0;

        {
            atcg::Timer timer;

            for(uint32_t i = 0; i <= steps_large; ++i)
            {
                u_explicit_large += delta_large * L * u_explicit_large;
            }

            std::cout << "Time: " << timer.elapsedSeconds() << "s\n"; 
        }

        {
            atcg::Timer timer;

            for(uint32_t i = 0; i <= steps_small; ++i)
            {
                u_explicit_small += delta_small * L * u_explicit_small;
            }

            std::cout << "Time: " << timer.elapsedSeconds() << "s\n"; 
        }

        Eigen::SparseMatrix<float> identity(L.cols(), L.rows());
        identity.setIdentity();

        {
            atcg::Timer timer;

            Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
            solver.compute(identity - delta_large * L);

            for(uint32_t i = 0; i <= steps_large; ++i)
            {
                u_implict_large = solver.solve(u_implict_large);
            }

            std::cout << "Time: " << timer.elapsedSeconds() << "s\n"; 
        }

        color_mesh(mesh_explicit_large, u_explicit_large);
        color_mesh(mesh_explicit_small, u_explicit_small);
        color_mesh(mesh_implicit_large, u_implict_large);

        mesh_explicit_large->uploadData();
        mesh_explicit_small->uploadData();
        mesh_implicit_large->uploadData();

        mesh = mesh_explicit_large;
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

        if(ImGui::BeginMenu("Exercise"))
        {
            ImGui::MenuItem("Show Diffusion Settings", nullptr, &show_diffusion);

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

        if(show_diffusion)
        {
            ImGui::Begin("Show Diffusion", &show_diffusion);
            
            if(ImGui::Button("Explicit large"))
            {
                mesh = mesh_explicit_large;
            }

            if(ImGui::Button("Explicit small"))
            {
                mesh = mesh_explicit_small;
            }

            if(ImGui::Button("Implicit large"))
            {
                mesh = mesh_implicit_large;
            }

            ImGui::End();
        }

    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event& event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
    }

private:
    std::shared_ptr<atcg::CameraController> camera_controller;
    std::shared_ptr<atcg::Mesh> mesh;

    std::shared_ptr<atcg::Mesh> mesh_explicit_large;
    std::shared_ptr<atcg::Mesh> mesh_explicit_small;
    std::shared_ptr<atcg::Mesh> mesh_implicit_large;

    atcg::Laplacian<float> laplacian;

    bool show_render_settings = false;
    bool show_diffusion = true;
    bool render_faces = true;
    bool render_points = false;
    bool render_edges = false;
};

class G06 : public atcg::Application
{
    public:

    G06()
        :atcg::Application()
    {
        pushLayer(new G06Layer("Layer"));
    }

    ~G06() {}

};

atcg::Application* atcg::createApplication()
{
    return new G06;
}