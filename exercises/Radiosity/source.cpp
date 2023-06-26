#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <ImGuizmo.h>

#include <glm/gtx/transform.hpp>

#include "Radiosity.h"

class RadiosityLayer : public atcg::Layer
{
public:
    RadiosityLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Renderer::setPointSize(2.0f);
        // atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = atcg::make_ref<atcg::FocusedController>(aspect_ratio);

        // TODO FIX RADIOSITY
        // mesh = atcg::IO::read_mesh("res/cornell_box_radiosity.ply", true);
        //   mesh->uploadData();

        /*Eigen::MatrixX3f emission = Eigen::MatrixX3f::Zero(mesh->n_faces(), 3);

        for(auto ft: mesh->faces())
        {
            glm::vec3 centroid = mesh->calc_centroid(ft);
            if(glm::length(centroid - glm::vec3(0, 10, 0)) < 1.0f)
            {
                emission.row(ft.idx()) = Eigen::Vector3f {50.0f, 50.0f, 50.0f};
            }
        }*/

        // mesh = solve_radiosity(mesh, emission);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        atcg::Renderer::draw(mesh,
                             camera_controller->getCamera(),
                             glm::mat4(1),
                             glm::vec3(1),
                             atcg::ShaderManager::getShader("flat"));
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

            ImGui::End();
        }
    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override { camera_controller->onEvent(event); }

private:
    atcg::ref_ptr<atcg::FocusedController> camera_controller;
    atcg::ref_ptr<atcg::Graph> mesh;

    bool show_render_settings = false;
};

class Radiosity : public atcg::Application
{
public:
    Radiosity() : atcg::Application() { pushLayer(new RadiosityLayer("Layer")); }

    ~Radiosity() {}
};

atcg::Application* atcg::createApplication()
{
    return new Radiosity;
}