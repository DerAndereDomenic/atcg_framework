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

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        shader = std::make_unique<atcg::Shader>("shader/base.vs", "shader/base.fs");

        float aspect_ratio = (float)atcg::Application::get()->getWindow()->getWidth() / (float)atcg::Application::get()->getWindow()->getHeight();
        camera_controller = std::make_unique<atcg::CameraController>(aspect_ratio);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        shader->use();
        const std::unique_ptr<atcg::Camera>& camera = camera_controller->getCamera();
        shader->setMVP(glm::mat4(1), camera->getView(), camera->getProjection());
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);

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
    unsigned int vbo;
    unsigned int vao;
    std::unique_ptr<atcg::Shader> shader;
    std::unique_ptr<atcg::CameraController> camera_controller;
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