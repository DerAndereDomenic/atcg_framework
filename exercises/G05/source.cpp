#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <glfw/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <queue>

#include <numeric>

using VertexHandle = atcg::Mesh::VertexHandle;
using EdgeHandle = atcg::Mesh::EdgeHandle;

template<typename T>
struct LaplaceUniform
{
    atcg::Laplacian<T> calculate(const std::shared_ptr<atcg::Mesh>& mesh)
    {
        std::vector<Eigen::Triplet<T>> edge_weights;

        for(auto e_it = mesh->edges_begin(); e_it != mesh->edges_end(); ++e_it)
        {
            uint32_t i = e_it->v0().idx();
            uint32_t j = e_it->v1().idx();

            edge_weights.emplace_back(i, j, 1.0);
            edge_weights.emplace_back(j, i, 1.0);
            edge_weights.emplace_back(i, i, -1.0);
            edge_weights.emplace_back(j, j, -1.0);
        }

        std::vector<Eigen::Triplet<T>> vertex_weights;

        for(auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end(); ++v_it)
        {
            vertex_weights.emplace_back(v_it->idx(), v_it->idx(), v_it->valence());
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

class G05Layer : public atcg::Layer
{
public:

    G05Layer(const std::string& name) : atcg::Layer(name) {}

    template<typename T, class LaplaceCalculator>
    void taubin_smoothing(const std::shared_ptr<atcg::Mesh>& mesh, LaplaceCalculator calculator)
    {
        T lambda = 0.1f;
        T mu = -0.11f; 

        atcg::Laplacian<T> laplacian = calculator.calculate(mesh);
        Eigen::SparseMatrix<T> Id(mesh->n_vertices(), mesh->n_vertices());
        Id.setIdentity();
        auto K = Id - laplacian.M.cwiseInverse() * laplacian.S;
        auto taubin_operator = (Id - mu * K) * (Id - lambda * K);

        Eigen::Matrix<T, -1, -1> v(mesh->n_vertices(), 3);

        for(auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end(); ++v_it)
        {
            atcg::Mesh::Point p = mesh->point(*v_it);
            v(v_it->idx(),0) = p[0];
            v(v_it->idx(),1) = p[1];
            v(v_it->idx(),2) = p[2];
        }

        v = taubin_operator * v;

        for(auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end(); ++v_it)
        {
            mesh->set_point(*v_it, atcg::Mesh::Point{v(v_it->idx(),0),
                                                     v(v_it->idx(),1),
                                                     v(v_it->idx(),2)});
        }

        mesh->uploadData();
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        mesh = std::make_shared<atcg::Mesh>();
        OpenMesh::IO::read_mesh(*mesh.get(), "res/suzanne_blender.obj");
        mesh->uploadData();

        default_mesh = std::make_shared<atcg::Mesh>();
        OpenMesh::IO::read_mesh(*default_mesh.get(), "res/suzanne_blender.obj");
        default_mesh->uploadData();
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        if(mesh && render_faces)
            atcg::Renderer::draw(mesh, atcg::ShaderManager::getShader("base"), camera_controller->getCamera());

        if(mesh && render_points)
            atcg::Renderer::drawPoints(mesh, glm::vec3(0), atcg::ShaderManager::getShader("flat"), camera_controller->getCamera());

        if(mesh && render_edges)
            atcg::Renderer::drawLines(mesh, glm::vec3(0), camera_controller->getCamera());
    }

    virtual void onImGuiRender() override
    {
        ImGui::BeginMainMenuBar();

        if(ImGui::BeginMenu("Rendering"))
        {
            ImGui::MenuItem("Show Render Settings", nullptr, &show_render_settings);

            ImGui::MenuItem("Show Taubin Smoothing", nullptr, &show_taubin);

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

        if(show_taubin)
        {
            ImGui::Begin("Taubin Smoothing", &show_taubin);

            bool update = ImGui::Checkbox("Use Cotan weights", &use_cotan);
            update = ImGui::InputInt("Number iterations", &num_iterations) || update;

            if(update)
            {
                mesh->copy_all_kernel_properties(*default_mesh.get());
                for(uint32_t n = 0; n < num_iterations; ++n)
                {
                    if(use_cotan)
                    {
                        taubin_smoothing<float, LaplaceCotan<float>>(mesh, LaplaceCotan<float>());
                    }
                    else
                    {
                        taubin_smoothing<float, LaplaceUniform<float>>(mesh, LaplaceUniform<float>());
                    }
                }
            }

            ImGui::End();
        }

    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event& event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
        dispatcher.dispatch<atcg::FileDroppedEvent>(ATCG_BIND_EVENT_FN(G05Layer::onFileDropped));
    }

    bool onFileDropped(atcg::FileDroppedEvent& event)
    {
        mesh = std::make_shared<atcg::Mesh>();
        OpenMesh::IO::read_mesh(*mesh.get(), event.getPath());
        mesh->uploadData();

        default_mesh = std::make_shared<atcg::Mesh>();
        OpenMesh::IO::read_mesh(*default_mesh.get(), event.getPath());
        default_mesh->uploadData();

        //Also reset camera
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        return true;
    }

private:
    std::shared_ptr<atcg::CameraController> camera_controller;
    std::shared_ptr<atcg::Mesh> mesh;
    std::shared_ptr<atcg::Mesh> default_mesh;

    bool show_render_settings = false;
    bool render_faces = true;
    bool render_points = false;
    bool render_edges = false;
    bool show_taubin = true;
    bool use_cotan = false;

    int num_iterations = 10;
};

class G05 : public atcg::Application
{
    public:

    G05()
        :atcg::Application()
    {
        pushLayer(new G05Layer("Layer"));
    }

    ~G05() {}

};

atcg::Application* atcg::createApplication()
{
    return new G05;
}