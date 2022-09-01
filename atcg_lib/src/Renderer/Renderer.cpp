#include <Renderer/Renderer.h>
#include <glad/glad.h>
#include <iostream>

namespace atcg
{
    Renderer* Renderer::s_renderer = new Renderer;

    void Renderer::initImpl()
    {
        if(!gladLoadGL())
        {
            std::cerr << "Error loading glad!\n";
        }
    }

    void Renderer::clearColorImpl(const glm::vec4& color)
    {
        glClearColor(color.r, color.g, color.b, color.a);
    }

    void Renderer::clearImpl()
    {
        glClear(GL_COLOR_BUFFER_BIT);
    }
}