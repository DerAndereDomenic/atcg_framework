#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <ImGuizmo.h>

#include <random>

#include "kernels.h"

class ClothLayer : public atcg::Layer
{
public:
    ClothLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = atcg::make_ref<atcg::FocusedController>(aspect_ratio);

        atcg::Renderer::setPointSize(0.25f);

        std::vector<atcg::Vertex> host_points;
        for(int i = 0; i < grid_size; ++i)
        {
            for(int j = 0; j < grid_size; ++j)
            {
                host_points.push_back(
                    {glm::vec3(-grid_size / 2 + j, -grid_size / 2 + i, 0.0f), glm::vec3(1), glm::vec3(1)});
            }
        }

        std::vector<atcg::Edge> edges;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distrib(0.0f, 1.0f);

        for(int i = 0; i < grid_size; ++i)
        {
            for(int j = 0; j < grid_size; ++j)
            {
                int dx = i - 1;
                if(!(dx < 0 || dx >= grid_size))
                {
                    edges.push_back({glm::vec2(i + grid_size * j, dx + grid_size * j),
                                     glm::vec3(distrib(gen), distrib(gen), distrib(gen)),
                                     0.1f});
                }

                int dy = j - 1;
                if(!(dy < 0 || dy >= grid_size))
                {
                    edges.push_back({glm::vec2(i + grid_size * j, i + grid_size * dy),
                                     glm::vec3(distrib(gen), distrib(gen), distrib(gen)),
                                     0.1f});
                }
            }
        }

        grid = atcg::Graph::createGraph(host_points, edges);

        plane = atcg::IO::read_mesh("res/plane_low.obj");

        checkerboard_shader =
            atcg::make_ref<atcg::Shader>("exercises/Cloth/checkerboard.vs", "exercises/Cloth/checkerboard.fs");
        checkerboard_shader->setFloat("checker_size", 0.1f);

        scene = atcg::make_ref<atcg::Scene>();

        atcg::Entity grid_entity = scene->createEntity();
        grid_entity.addComponent<atcg::GeometryComponent>(grid)
            .addConfig(
                {atcg::ShaderManager::getShader("base"), glm::vec3(1), atcg::DrawMode::ATCG_DRAW_MODE_EDGES_CYLINDER})
            .addConfig(
                {atcg::ShaderManager::getShader("base"), glm::vec3(1), atcg::DrawMode::ATCG_DRAW_MODE_POINTS_SPHERE});
        grid_entity.addComponent<atcg::TransformComponent>();

        atcg::Entity plane_entity = scene->createEntity();
        plane_entity.addComponent<atcg::GeometryComponent>(plane).addConfig({checkerboard_shader});
        auto& transform = plane_entity.addComponent<atcg::TransformComponent>();
        transform.setScale(glm::vec3(100, 100, 100));
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        // ATCG_TRACE("{0} s | {1} fps", delta_time, 1.0f / delta_time);
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        time += delta_time;

        atcg::Vertex* dev_ptr = grid->getVerticesBuffer()->getDevicePointer<atcg::Vertex>();

        simulate(dev_ptr, grid_size * grid_size, time);

        atcg::Renderer::draw(scene, camera_controller->getCamera());
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

        if(hovered_entity)
        {
            ImGuizmo::SetOrthographic(false);
            ImGuizmo::BeginFrame();

            const auto& window   = atcg::Application::get()->getWindow();
            glm::vec2 window_pos = window->getPosition();
            ImGuizmo::SetRect(window_pos.x, window_pos.y, (float)window->getWidth(), (float)window->getHeight());

            glm::mat4 camera_projection = camera_controller->getCamera()->getProjection();
            glm::mat4 camera_view       = camera_controller->getCamera()->getView();

            atcg::TransformComponent& transform = hovered_entity.getComponent<atcg::TransformComponent>();
            glm::mat4 model                     = transform.getModel();

            ImGuizmo::Manipulate(glm::value_ptr(camera_view),
                                 glm::value_ptr(camera_projection),
                                 current_operation,
                                 ImGuizmo::LOCAL,
                                 glm::value_ptr(model));
            transform.setModel(model);
        }
    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
        dispatcher.dispatch<atcg::MouseMovedEvent>(ATCG_BIND_EVENT_FN(ClothLayer::onMouseMoved));
        dispatcher.dispatch<atcg::MouseButtonPressedEvent>(ATCG_BIND_EVENT_FN(ClothLayer::onMousePressed));
        dispatcher.dispatch<atcg::KeyPressedEvent>(ATCG_BIND_EVENT_FN(ClothLayer::onKeyPressed));
    }

    bool onKeyPressed(atcg::KeyPressedEvent* event)
    {
        if(event->getKeyCode() == GLFW_KEY_T) { current_operation = ImGuizmo::OPERATION::TRANSLATE; }
        if(event->getKeyCode() == GLFW_KEY_R) { current_operation = ImGuizmo::OPERATION::ROTATE; }
        if(event->getKeyCode() == GLFW_KEY_S) { current_operation = ImGuizmo::OPERATION::SCALE; }
        // if(event->getKeyCode() == GLFW_KEY_L) { camera_controller->getCamera()->setLookAt(sphere->getPosition()); }

        return true;
    }

    bool onMousePressed(atcg::MouseButtonPressedEvent* event)
    {
        if(event->getMouseButton() == GLFW_MOUSE_BUTTON_LEFT)
        {
            int id         = atcg::Renderer::getEntityIndex(mouse_pos);
            hovered_entity = id == -1 ? atcg::Entity() : atcg::Entity((entt::entity)id, scene.get());
        }
        return true;
    }

    bool onMouseMoved(atcg::MouseMovedEvent* event)
    {
        const auto& window = atcg::Application::get()->getWindow();
        mouse_pos          = glm::vec2(event->getX(), window->getHeight() - event->getY());

        return false;
    }

private:
    atcg::ref_ptr<atcg::Scene> scene;
    atcg::Entity hovered_entity;

    atcg::ref_ptr<atcg::FirstPersonController> camera_controller;
    atcg::ref_ptr<atcg::Graph> grid;
    int32_t grid_size = 51;

    atcg::ref_ptr<atcg::Graph> plane;
    atcg::ref_ptr<atcg::Shader> checkerboard_shader;

    float time = 0.0f;

    glm::vec2 mouse_pos;

    bool show_render_settings             = false;
    ImGuizmo::OPERATION current_operation = ImGuizmo::OPERATION::TRANSLATE;
};

class Cloth : public atcg::Application
{
public:
    Cloth() : atcg::Application() { pushLayer(new ClothLayer("Layer")); }

    ~Cloth() {}
};

atcg::Application* atcg::createApplication()
{
    return new Cloth;
}