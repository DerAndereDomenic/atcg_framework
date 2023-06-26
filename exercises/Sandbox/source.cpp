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

        cube = atcg::IO::read_mesh("res/cube.obj");

        atcg::ShaderManager::addShaderFromName("volume");

        noise_texture = atcg::Noise::createWorleyNoiseTexture3D(glm::ivec3(128), num_points);

        scene       = atcg::make_ref<atcg::Scene>();
        cube_entity = scene->createEntity();
        cube_entity.addComponent<atcg::TransformComponent>();
        cube_entity.addComponent<atcg::GeometryComponent>(cube).addConfig(
            {atcg::ShaderManager::getShader("volume"), glm::vec3(1), atcg::DrawMode::ATCG_DRAW_MODE_TRIANGLE});

        light_entity = scene->createEntity();
        light_entity.addComponent<atcg::TransformComponent>();

        selected_entity = light_entity;
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();


        glm::vec3 light_pos = light_entity.getComponent<atcg::TransformComponent>().getPosition();
        atcg::ShaderManager::getShader("volume")->setInt("noise_texture", 0);
        atcg::ShaderManager::getShader("volume")->setVec3("light_position", light_pos);
        atcg::ShaderManager::getShader("volume")->setFloat("sigma_s_base", sigma_s_base);
        atcg::ShaderManager::getShader("volume")->setFloat("sigma_a_base", sigma_a_base);
        atcg::ShaderManager::getShader("volume")->setFloat("g", g);
        noise_texture->use();
        atcg::Renderer::draw(cube_entity, camera_controller->getCamera());
        dt = delta_time;
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

            std::stringstream ss;
            ss << "FPS: " << 1.0f / dt << " | " << dt << " ms\n";
            ImGui::Text(ss.str().c_str());

            if(ImGui::SliderInt("Number of points", reinterpret_cast<int*>(&num_points), 1, 512))
            {
                noise_texture = atcg::Noise::createWorleyNoiseTexture3D(glm::ivec3(128), num_points);
            }

            ImGui::InputFloat("sigma_s_base", &sigma_s_base);

            ImGui::InputFloat("sigma_a_base", &sigma_a_base);

            ImGui::SliderFloat("g", &g, -0.9999f, 0.9999f);

            ImGui::End();
        }

        ImGui::Begin("Scene Panel");

        if(ImGui::Button("Light")) { selected_entity = light_entity; }

        if(ImGui::Button("Volume")) { selected_entity = cube_entity; }

        ImGui::End();

        // Gizmo test
        ImGuizmo::SetOrthographic(false);
        ImGuizmo::BeginFrame();

        const auto& window   = atcg::Application::get()->getWindow();
        glm::vec2 window_pos = window->getPosition();
        ImGuizmo::SetRect(window_pos.x, window_pos.y, (float)window->getWidth(), (float)window->getHeight());

        glm::mat4 camera_projection = camera_controller->getCamera()->getProjection();
        glm::mat4 camera_view       = camera_controller->getCamera()->getView();

        glm::mat4 transform =
            selected_entity.getComponent<atcg::TransformComponent>().getModel();    // sphere->getModel();

        ImGuizmo::Manipulate(glm::value_ptr(camera_view),
                             glm::value_ptr(camera_projection),
                             current_operation,
                             ImGuizmo::LOCAL,
                             glm::value_ptr(transform));

        selected_entity.getComponent<atcg::TransformComponent>().setModel(transform);

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
        // if(event->getKeyCode() == GLFW_KEY_L) { camera_controller->getCamera()->setLookAt(sphere->getPosition()); }

        return true;
    }

private:
    atcg::ref_ptr<atcg::Scene> scene;
    atcg::Entity cube_entity;
    atcg::Entity light_entity;
    atcg::Entity selected_entity;
    atcg::ref_ptr<atcg::FocusedController> camera_controller;
    atcg::ref_ptr<atcg::Graph> cube;

    atcg::ref_ptr<atcg::Texture3D> noise_texture;

    uint32_t num_points = 16;

    bool show_render_settings = true;

    float sigma_s_base = 20.0f;
    float sigma_a_base = 0.0f;
    float g            = 0.0f;
    float dt           = 1.0f / 60.0f;

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