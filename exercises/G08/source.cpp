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

    T triangleCotan(const atcg::TriMesh::Point& v0, const atcg::TriMesh::Point& v1, const atcg::TriMesh::Point& v2)
    {
        const auto d0 = v0 - v2;
        const auto d1 = v1 - v2;
        const auto d2 = v1 - v0;
        const auto area = atcg::areaFromMetric<T>(d0.norm(), d1.norm(), d2.norm());
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

class G08Layer : public atcg::Layer
{
public:

    G08Layer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        mesh = std::make_shared<atcg::Mesh>();
        OpenMesh::IO::read_mesh(*mesh.get(), "res/bunny.obj");
        mesh->request_vertex_colors();
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

        atcg::EventDispatcher dispatcher(event);
    }

private:
    std::shared_ptr<atcg::CameraController> camera_controller;
    std::shared_ptr<atcg::Mesh> mesh;

    bool show_render_settings = false;
    bool render_faces = true;
    bool render_points = false;
    bool render_edges = false;
};

class G08 : public atcg::Application
{
    public:

    G08()
        :atcg::Application()
    {
        pushLayer(new G08Layer("Layer"));
    }

    ~G08() {}

};

atcg::Application* atcg::createApplication()
{
    return new G08;
}