#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <glfw/glfw3.h>
#include <imgui.h>

class G01Layer : public atcg::Layer
{
public:

    G01Layer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        std::shared_ptr<atcg::TriMesh> mesh = std::make_shared<atcg::TriMesh>();
        OpenMesh::IO::read_mesh(*mesh.get(), "res/suzanne_blender.obj");

        render_mesh = std::make_shared<atcg::RenderMesh>();
        render_mesh->uploadData(mesh);

        //render_mesh->setScale(glm::vec3(0.7f));
        //render_mesh->setRotation(glm::vec3(-1,0,1), glm::radians(210.0f));
        //render_mesh->setPosition(glm::vec3(0,0,3));

        float aspect_ratio = (float)atcg::Application::get()->getWindow()->getWidth() / (float)atcg::Application::get()->getWindow()->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        atcg::Renderer::draw(render_mesh, atcg::ShaderManager::getShader("base"), camera_controller->getCamera());
        atcg::Renderer::drawPoints(render_mesh, glm::vec3(1,0,0), atcg::ShaderManager::getShader("flat"), camera_controller->getCamera());

        if(atcg::Input::isKeyPressed(GLFW_KEY_SPACE))
        {
            std::cout << "Pressed Space!\n";
        }
    }

    virtual void onImGuiRender() override
    {
        ImGui::BeginMainMenuBar();

        if(ImGui::BeginMenu("Exercise"))
        {
            ImGui::MenuItem("Test", nullptr, &show_test_window);
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();

        if(show_test_window)
        {
            ImGui::Begin("Test", &show_test_window);
            ImGui::End();
        }

    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event& event) override
    {
        camera_controller->onEvent(event);
    }

private:
    std::shared_ptr<atcg::RenderMesh> render_mesh;
    std::shared_ptr<atcg::CameraController> camera_controller;
    bool show_test_window = false;
};

class G01 : public atcg::Application
{
    public:

    G01()
        :atcg::Application()
    {
        pushLayer(new G01Layer("Layer"));
    }

    ~G01() {}

};

atcg::Application* atcg::createApplication()
{
    return new G01;
}