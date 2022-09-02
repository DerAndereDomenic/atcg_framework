#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <glfw/glfw3.h>

class G01Layer : public atcg::Layer
{
public:

    G01Layer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        float vertices[] = 
        {
            -0.5f, -0.5f, 0.0f,
            0.5f, -0.5f, 0.0f,
            0.0f, 0.5f, 0.0f
        };

        uint32_t indices[] = 
        {
            0, 1, 2
        };

        vao = std::make_shared<atcg::VertexArray>();
        vbo = std::make_shared<atcg::VertexBuffer>(vertices, static_cast<uint32_t>(sizeof(vertices)));
        vbo->setLayout({{atcg::ShaderDataType::Float3, "aPosition"}});

        vao->addVertexBuffer(vbo);

        std::shared_ptr<atcg::IndexBuffer> ibo = std::make_shared<atcg::IndexBuffer>(indices, 3);
        vao->setIndexBuffer(ibo);

        shader = std::make_shared<atcg::Shader>("shader/base.vs", "shader/base.fs");

        float aspect_ratio = (float)atcg::Application::get()->getWindow()->getWidth() / (float)atcg::Application::get()->getWindow()->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        atcg::Renderer::draw(vao, shader, camera_controller->getCamera());

        if(atcg::Input::isKeyPressed(GLFW_KEY_SPACE))
        {
            std::cout << "Pressed Space!\n";
        }
    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event& event) override
    {
        camera_controller->onEvent(event);
    }

private:
    std::shared_ptr<atcg::VertexArray> vao;
    std::shared_ptr<atcg::VertexBuffer> vbo;
    std::shared_ptr<atcg::Shader> shader;
    std::shared_ptr<atcg::CameraController> camera_controller;
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