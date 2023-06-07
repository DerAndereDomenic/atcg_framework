#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glm/gtc/type_ptr.hpp>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <ImGuizmo.h>

#include <glm/gtx/transform.hpp>

#include <random>

class ClothLayer : public atcg::Layer
{
public:
    ClothLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = atcg::make_ref<atcg::CameraController>(aspect_ratio);

        std::vector<glm::vec3> host_points;
        for(int i = 0; i < grid_size; ++i)
        {
            for(int j = 0; j < grid_size; ++j)
            {
                host_points.push_back(glm::vec3(-grid_size / 2 + j, -grid_size / 2 + i, 0.0f));
            }
        }

        points = atcg::ref_ptr<glm::vec3>(host_points.size());
        points.upload(host_points.data());

        std::vector<float> grid;

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
                    grid.push_back(i + grid_size * j);
                    grid.push_back(dx + grid_size * j);
                    grid.push_back(distrib(gen));
                    grid.push_back(distrib(gen));
                    grid.push_back(distrib(gen));
                }

                int dy = j - 1;
                if(!(dy < 0 || dy >= grid_size))
                {
                    grid.push_back(i + grid_size * j);
                    grid.push_back(i + grid_size * dy);
                    grid.push_back(distrib(gen));
                    grid.push_back(distrib(gen));
                    grid.push_back(distrib(gen));
                }
            }
        }

        points_vbo = atcg::make_ref<atcg::VertexBuffer>((void*)points.get(), points.size() * sizeof(glm::vec3));

        grid_vbo = atcg::make_ref<atcg::VertexBuffer>((void*)grid.data(), grid.size() * sizeof(uint32_t));
        grid_vbo->setLayout({{atcg::ShaderDataType::Float2, "aIndex"}, {atcg::ShaderDataType::Float3, "aColor"}});
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        time += delta_time;

        for(int i = 0; i < grid_size; ++i)
        {
            for(int j = 0; j < grid_size; ++j)
            {
                points.get()[i + grid_size * j].z = glm::sin(2.0f * glm::pi<float>() * (time) + j / 3.0f + i);
            }
        }

        points_vbo->setData((void*)points.get(), sizeof(glm::vec3) * points.size());

        atcg::Renderer::drawGrid(points_vbo, grid_vbo, camera_controller->getCamera(), glm::vec3(1));
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
    virtual void onEvent(atcg::Event* event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
    }

private:
    atcg::ref_ptr<atcg::CameraController> camera_controller;
    atcg::ref_ptr<atcg::VertexBuffer> points_vbo;
    atcg::ref_ptr<atcg::VertexBuffer> grid_vbo;
    atcg::ref_ptr<glm::vec3> points;
    int32_t grid_size = 51;

    float time = 0.0f;

    bool show_render_settings = false;
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