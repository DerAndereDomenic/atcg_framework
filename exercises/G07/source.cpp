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

class G07Layer : public atcg::Layer
{
public:

    G07Layer(const std::string& name) : atcg::Layer(name) {}

    void detect_boundary_edges(const std::shared_ptr<atcg::Mesh>& mesh, std::vector<EdgeHandle>& boundary_edges)
    {
        for(auto e_it = mesh->edges_begin(); e_it != mesh->edges_end(); ++e_it)
        {
            if(mesh->is_boundary(*e_it))
            {
                boundary_edges.push_back(*e_it);
            }
        }
    }

    std::vector<VertexHandle> detect_boundary_path(const std::shared_ptr<atcg::Mesh>& mesh, const std::vector<EdgeHandle>& boundary_edges)
    {
        std::vector<VertexHandle> boundary_path;
        
        VertexHandle start = mesh->from_vertex_handle(mesh->halfedge_handle(boundary_edges[0], 0));
        VertexHandle current_to = mesh->to_vertex_handle(mesh->halfedge_handle(boundary_edges[0], 0));
        boundary_path.push_back(start);
        boundary_path.push_back(current_to);

        for(uint32_t i = 1; i < boundary_edges.size(); ++i)
        {
            for(uint32_t j = 1; j < boundary_edges.size(); ++j)
            {
                VertexHandle from = mesh->from_vertex_handle(mesh->halfedge_handle(boundary_edges[j], 0));
                if(from.idx() == current_to.idx())
                {
                    current_to = mesh->to_vertex_handle(mesh->halfedge_handle(boundary_edges[j], 0));
                    boundary_path.push_back(current_to);
                    break;
                }
            }
        }

        return boundary_path;
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        mesh = std::make_shared<atcg::Mesh>();
        OpenMesh::IO::read_mesh(*mesh.get(), "res/maxear.obj");
        mesh->request_vertex_colors();

        float max_scale = -std::numeric_limits<float>::infinity();
        atcg::TriMesh::Point mean_point{0,0,0};
        for(auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end(); ++v_it)
        {
            for(uint32_t i = 0; i < 3; ++i)
            {
                if(mesh->point(*v_it)[i] > max_scale)
                {
                    max_scale = mesh->point(*v_it)[i];
                }
            }

            mean_point += mesh->point(*v_it);
        }
        mean_point /= static_cast<float>(mesh->n_vertices());

        for(auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end(); ++v_it)
        {
            mesh->set_point(*v_it, (mesh->point(*v_it) - mean_point) / max_scale);
        }

        std::vector<EdgeHandle> boundary_edges;
        detect_boundary_edges(mesh, boundary_edges);
        std::vector<VertexHandle> boundary_path = detect_boundary_path(mesh, boundary_edges);

        int index = 0;
        for(auto v : boundary_path)
        {
            std::cout << v.idx() << std::endl;
            uint8_t color = static_cast<uint8_t>(static_cast<float>(index)/static_cast<float>(boundary_path.size()) * 255.0f);
            mesh->set_color(v, atcg::Mesh::Color{color,0,0});
            ++index;
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
            atcg::Renderer::drawLines(mesh, glm::vec3(0), camera_controller->getCamera());
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

class G07 : public atcg::Application
{
    public:

    G07()
        :atcg::Application()
    {
        pushLayer(new G07Layer("Layer"));
    }

    ~G07() {}

};

atcg::Application* atcg::createApplication()
{
    return new G07;
}