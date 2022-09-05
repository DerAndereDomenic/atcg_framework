#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <glfw/glfw3.h>
#include <imgui.h>
#include <algorithm>

class G04Layer : public atcg::Layer
{
public:

    G04Layer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        mesh = std::make_shared<atcg::TriMesh>();
        OpenMesh::IO::read_mesh(*mesh.get(), "res/bunny.obj");

        render_mesh = std::make_shared<atcg::RenderMesh>();
        render_mesh->uploadData(mesh);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        if(render_mesh && render_faces)
            atcg::Renderer::draw(render_mesh, atcg::ShaderManager::getShader("base"), camera_controller->getCamera());

        if(render_mesh && render_points)
            atcg::Renderer::drawPoints(render_mesh, glm::vec3(0), atcg::ShaderManager::getShader("flat"), camera_controller->getCamera());

        if(render_mesh && render_edges)
            atcg::Renderer::drawLines(render_mesh, glm::vec3(0), camera_controller->getCamera());
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
        dispatcher.dispatch<atcg::FileDroppedEvent>(ATCG_BIND_EVENT_FN(G04Layer::onFileDropped));
    }

    bool onFileDropped(atcg::FileDroppedEvent& event)
    {
        mesh = std::make_shared<atcg::TriMesh>();
        OpenMesh::IO::read_mesh(*mesh.get(), event.getPath());

        render_mesh = std::make_shared<atcg::RenderMesh>();
        render_mesh->uploadData(mesh);

        //Also reset camera
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        return true;
    }

private:
    std::shared_ptr<atcg::TriMesh> mesh;
    std::shared_ptr<atcg::RenderMesh> render_mesh;
    std::shared_ptr<atcg::CameraController> camera_controller;

    bool show_render_settings = false;
    bool render_faces = true;
    bool render_points = false;
    bool render_edges = false;
};

class G04 : public atcg::Application
{
    public:

    G04()
        :atcg::Application()
    {
        pushLayer(new G04Layer("Layer"));
    }

    ~G04() {}

};

atcg::Application* atcg::createApplication()
{
    return new G04;
}