#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <ImGuizmo.h>


class SandboxLayer : public atcg::Layer
{
public:
    SandboxLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Renderer::setPointSize(2.0f);
        atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = atcg::make_ref<atcg::FocusedController>(aspect_ratio);

        sphere = atcg::IO::read_mesh("res/cube.obj");
        sphere->uploadData();

        atcg::ShaderManager::addShaderFromName("volume");

        noise_texture = atcg::Noise::createWorleyNoiseTexture3D(glm::ivec3(128), num_points);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();


        atcg::ShaderManager::getShader("volume")->setInt("noise_texture", 0);
        atcg::ShaderManager::getShader("volume")->setVec3("light_position", light_pos);
        noise_texture->use();
        atcg::Renderer::draw(sphere,
                             camera_controller->getCamera(),
                             glm::vec3(1),
                             atcg::ShaderManager::getShader("volume"));
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

            if(ImGui::SliderInt("Number of points", reinterpret_cast<int*>(&num_points), 1, 512))
            {
                noise_texture = atcg::Noise::createWorleyNoiseTexture3D(glm::ivec3(128), num_points);
            }

            ImGui::End();
        }

        // Gizmo test
        ImGuizmo::SetOrthographic(false);
        ImGuizmo::BeginFrame();

        const auto& window   = atcg::Application::get()->getWindow();
        glm::vec2 window_pos = window->getPosition();
        ImGuizmo::SetRect(window_pos.x, window_pos.y, (float)window->getWidth(), (float)window->getHeight());

        glm::mat4 camera_projection = camera_controller->getCamera()->getProjection();
        glm::mat4 camera_view       = camera_controller->getCamera()->getView();

        glm::mat4 transform = glm::translate(light_pos);    // sphere->getModel();

        ImGuizmo::Manipulate(glm::value_ptr(camera_view),
                             glm::value_ptr(camera_projection),
                             current_operation,
                             ImGuizmo::LOCAL,
                             glm::value_ptr(transform));

        light_pos = transform[3];

        // if(ImGuizmo::IsUsing()) { sphere->setModel(transform); }
    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
        dispatcher.dispatch<atcg::KeyPressedEvent>(ATCG_BIND_EVENT_FN(SandboxLayer::onKeyPressed));
    }

    bool onKeyPressed(atcg::KeyPressedEvent* event)
    {
        if(event->getKeyCode() == GLFW_KEY_T) { current_operation = ImGuizmo::OPERATION::TRANSLATE; }
        if(event->getKeyCode() == GLFW_KEY_R) { current_operation = ImGuizmo::OPERATION::ROTATE; }
        if(event->getKeyCode() == GLFW_KEY_S) { current_operation = ImGuizmo::OPERATION::SCALE; }
        if(event->getKeyCode() == GLFW_KEY_L) { camera_controller->getCamera()->setLookAt(sphere->getPosition()); }

        return true;
    }

private:
    atcg::ref_ptr<atcg::FocusedController> camera_controller;
    atcg::ref_ptr<atcg::Mesh> sphere;

    atcg::ref_ptr<atcg::Texture3D> noise_texture;

    uint32_t num_points = 16;
    glm::vec3 light_pos = glm::vec3(0);

    bool show_render_settings = true;

    ImGuizmo::OPERATION current_operation = ImGuizmo::OPERATION::TRANSLATE;
};

class Sandbox : public atcg::Application
{
public:
    Sandbox() : atcg::Application() { pushLayer(new SandboxLayer("Layer")); }

    ~Sandbox() {}
};

atcg::Application* atcg::createApplication()
{
    return new Sandbox;
}