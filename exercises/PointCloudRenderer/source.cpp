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
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        for(auto it = clouds.begin(); it != clouds.end(); ++it)
        {
            if(it->second)
                atcg::Renderer::draw(it->first, atcg::ShaderManager::getShader("flat"), camera_controller->getCamera());
        }
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
    virtual void onEvent(atcg::Event& event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
        dispatcher.dispatch<atcg::FileDroppedEvent>(ATCG_BIND_EVENT_FN(PointCloudLayer::onFileDropped));
    }

    bool onFileDropped(atcg::FileDroppedEvent& event)
    {
        auto point_cloud = atcg::IO::read_pointcloud(event.getPath().c_str());
        atcg::normalize(point_cloud);
        point_cloud->uploadData();
        clouds.push_back(std::make_pair(point_cloud, true));

        //Also reset camera
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        return true;
    }

private:
    using CloudList = std::vector<std::pair<std::shared_ptr<atcg::PointCloud>,bool>>;

    CloudList clouds;
    std::shared_ptr<atcg::CameraController> camera_controller;

    bool show_render_settings = false;
};

class PointCloudRenderer : public atcg::Application
{
    public:

    PointCloudRenderer()
        :atcg::Application()
    {
        pushLayer(new PointCloudLayer("Layer"));
    }

    ~PointCloudRenderer() {}

};

atcg::Application* atcg::createApplication()
{
    return new PointCloudRenderer;
}