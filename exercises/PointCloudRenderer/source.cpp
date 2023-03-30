#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>

class PointCloudLayer : public atcg::Layer
{
public:
    PointCloudLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Renderer::setClearColor(glm::vec4(1));
        atcg::Renderer::setPointSize(2.0f);
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = std::make_shared<atcg::CameraController>(aspect_ratio);

        depth_values.resize(search_radius * search_radius);

        sphere = atcg::IO::read_mesh("res/sphere.obj");
        sphere->setScale(glm::vec3(0.01f));
        sphere->uploadData();

        {
            auto point_cloud = atcg::IO::read_pointcloud("C:/Users/zingsheim/Documents/PointCloudCompression/"
                                                         "sample.xyz");
            // auto point_cloud = atcg::IO::read_pointcloud("res/bunny.obj");
            point_cloud->uploadData();
            clouds.push_back(std::make_pair(point_cloud, true));
        }
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        for(auto it = clouds.begin(); it != clouds.end(); ++it)
        {
            if(it->second)
                atcg::Renderer::draw(it->first,
                                     camera_controller->getCamera(),
                                     glm::vec3(1),
                                     atcg::ShaderManager::getShader("flat"));
        }

        glReadPixels(static_cast<int>(mouse_pos.x - search_radius / 2),
                     static_cast<int>(mouse_pos.y - search_radius / 2),
                     search_radius,
                     search_radius,
                     GL_DEPTH_COMPONENT,
                     GL_FLOAT,
                     depth_values.data());
        float min = 1.0f;

        for(float depth: depth_values) { min = std::min(depth, min); }

        if(min != 0.0f && min != 1.0f)
        {
            // Project and render sphere

            const auto& window = atcg::Application::get()->getWindow();

            float width  = (float)window->getWidth();
            float height = (float)window->getHeight();

            float x_ndc = (static_cast<float>(mouse_pos.x)) / (static_cast<float>(width) / 2.0f) - 1.0f;
            float y_ndc = (static_cast<float>(mouse_pos.y)) / (static_cast<float>(height) / 2.0f) - 1.0f;

            glm::vec4 world_pos(x_ndc, y_ndc, 2 * min - 1, 1.0f);

            world_pos = glm::inverse(camera_controller->getCamera()->getViewProjection()) * world_pos;
            world_pos /= world_pos.w;

            if(atcg::Input::isKeyPressed(GLFW_KEY_P)) { sphere_pos.push_back(world_pos); }
        }

        // glDepthMask(false);
        for(glm::vec3 p: sphere_pos)
        {
            sphere->setPosition(p);
            atcg::Renderer::draw(sphere, camera_controller->getCamera());
        }
        // glDepthMask(true);
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

            int32_t id = 0;
            for(auto it = clouds.begin(); it != clouds.end(); ++it)
            {
                ImGui::PushID(id);
                ImGui::Checkbox("Render Cloud:", &(it->second));
                ImGui::PopID();
                ++id;
            }

            ImGui::End();
        }
    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
        dispatcher.dispatch<atcg::FileDroppedEvent>(ATCG_BIND_EVENT_FN(PointCloudLayer::onFileDropped));
        dispatcher.dispatch<atcg::MouseMovedEvent>(ATCG_BIND_EVENT_FN(PointCloudLayer::onMouseMoved));
    }

    bool onFileDropped(atcg::FileDroppedEvent* event)
    {
        auto point_cloud = atcg::IO::read_pointcloud(event->getPath().c_str());
        atcg::normalize(point_cloud);
        point_cloud->uploadData();
        clouds.push_back(std::make_pair(point_cloud, true));

        // Also reset camera
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = std::make_shared<atcg::CameraController>(aspect_ratio);

        return true;
    }

    bool onMouseMoved(atcg::MouseMovedEvent* event)
    {
        const auto& window = atcg::Application::get()->getWindow();
        mouse_pos          = glm::vec2(event->getX(), window->getHeight() - event->getY());

        return false;
    }

private:
    using CloudList = std::vector<std::pair<std::shared_ptr<atcg::PointCloud>, bool>>;

    CloudList clouds;
    std::shared_ptr<atcg::CameraController> camera_controller;
    std::shared_ptr<atcg::Mesh> sphere;
    std::vector<glm::vec3> sphere_pos;

    glm::vec2 mouse_pos;

    std::vector<float> depth_values;
    uint32_t search_radius = 10;

    bool show_render_settings = false;
};

class PointCloudRenderer : public atcg::Application
{
public:
    PointCloudRenderer() : atcg::Application() { pushLayer(new PointCloudLayer("Layer")); }

    ~PointCloudRenderer() {}
};

atcg::Application* atcg::createApplication()
{
    return new PointCloudRenderer;
}