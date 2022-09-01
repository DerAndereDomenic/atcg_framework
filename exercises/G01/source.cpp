#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

class G01Layer : public atcg::Layer
{
    public:

    G01Layer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Renderer::clearColor(glm::vec4(1,0,0,1));

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
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        atcg::Renderer::clear();

        shader->use();
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event& event) override
    {
        
    }

    private:
    unsigned int vbo;
    unsigned int vao;
    std::unique_ptr<atcg::Shader> shader;
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