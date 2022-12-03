#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>

class CloudLayer : public atcg::Layer
{
public:

    CloudLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Renderer::setPointSize(2.0f);
        atcg::ShaderManager::addShaderFromName("cloud");
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        cube = atcg::IO::read_mesh("res/cube.obj");
        cube->uploadData();
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        atcg::Renderer::draw(cube, atcg::ShaderManager::getShader("cloud"), camera_controller->getCamera());
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
    virtual void onEvent(atcg::Event& event) override
    {
        camera_controller->onEvent(event);
    }

private:
    std::shared_ptr<atcg::CameraController> camera_controller;
    std::shared_ptr<atcg::Mesh> cube;

    bool show_render_settings = false;
    bool show_cpd_settings = true;
};

class CloudRenderer : public atcg::Application
{
    public:

    CloudRenderer()
        :atcg::Application()
    {
        pushLayer(new CloudLayer("Layer"));
    }

    ~CloudRenderer() {}

};

atcg::Application* atcg::createApplication()
{
    return new CloudRenderer;
}