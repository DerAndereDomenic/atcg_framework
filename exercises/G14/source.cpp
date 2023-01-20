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

#include <glm/gtc/matrix_transform.hpp>

class G14Layer : public atcg::Layer
{
public:
    G14Layer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        mesh_source = atcg::IO::read_mesh("res/suzanne_blender.obj");
        mesh_target = atcg::IO::read_mesh("res/suzanne_blender.obj");

        mesh_source->request_vertex_colors();
        mesh_target->request_vertex_colors();

        for(uint32_t i = 0; i < mesh_target->n_vertices(); ++i)
        {
            mesh_source->set_color(atcg::Mesh::VertexHandle(i), atcg::Mesh::Color{255, 0, 0});
            mesh_target->set_color(atcg::Mesh::VertexHandle(i), atcg::Mesh::Color{0, 255, 0});

            atcg::Mesh::Point p_ = mesh_source->point(atcg::Mesh::VertexHandle(i));
            glm::vec4 p(p_[0], p_[1], p_[2], 1.0);

            glm::mat4 T = glm::translate(glm::rotate(glm::mat4(1), glm::pi<float>()/8.0f, glm::normalize(glm::vec3(0.0f, -0.1f, 0.8f))), glm::vec3(1.0f, -0.2f, 0.5f));
            p = T*p;

            mesh_source->set_point(atcg::Mesh::VertexHandle(i), {p.x, p.y, p.z});
        }

        mesh_source->uploadData();
        mesh_target->uploadData();
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        if(mesh_target && render_faces_target)
            atcg::Renderer::draw(mesh_target, atcg::ShaderManager::getShader("base"), camera_controller->getCamera());

        if(mesh_target && render_points_target)
            atcg::Renderer::drawPoints(mesh_target, glm::vec3(0), atcg::ShaderManager::getShader("base"), camera_controller->getCamera());

        if(mesh_target && render_edges_target)
            atcg::Renderer::drawLines(mesh_target, glm::vec3(1), camera_controller->getCamera());

        if(mesh_source && render_faces_source)
            atcg::Renderer::draw(mesh_source, atcg::ShaderManager::getShader("base"), camera_controller->getCamera());

        if(mesh_source && render_points_source)
            atcg::Renderer::drawPoints(mesh_source, glm::vec3(0), atcg::ShaderManager::getShader("base"), camera_controller->getCamera());

        if(mesh_source && render_edges_source)
            atcg::Renderer::drawLines(mesh_source, glm::vec3(1), camera_controller->getCamera());
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

            ImGui::Checkbox("Render Target Vertices", &render_points_target);
            ImGui::Checkbox("Render Target Edges", &render_edges_target);
            ImGui::Checkbox("Render Target Mesh", &render_faces_target);

            ImGui::Checkbox("Render Source Vertices", &render_points_source);
            ImGui::Checkbox("Render Source Edges", &render_edges_source);
            ImGui::Checkbox("Render Source Mesh", &render_faces_source);
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
    std::shared_ptr<atcg::Mesh> mesh_target;
    std::shared_ptr<atcg::Mesh> mesh_source;

    bool show_render_settings = true;
    bool render_faces_target = true;
    bool render_points_target = false;
    bool render_edges_target = false;

    bool render_faces_source = true;
    bool render_points_source = false;
    bool render_edges_source = false;

    bool show_edit_settings = true;
    float ks = 1.0;
    float kb = 1.0;
    float edit_radius = 0.3;
    float region_radius = 0.8;
    float edit_height = 1.0;
};

class G14 : public atcg::Application
{
    public:

    G14()
        :atcg::Application()
    {
        pushLayer(new G14Layer("Layer"));
    }

    ~G14() {}

};

atcg::Application* atcg::createApplication()
{
    return new G14;
}