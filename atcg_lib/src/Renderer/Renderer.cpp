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

    void Renderer::setViewportImpl(const uint32_t& x, const uint32_t& y, const uint32_t& width, const uint32_t& height)
    {
        glViewport(x, y, width, height);
    }

    void Renderer::clearImpl()
    {
        glClear(GL_COLOR_BUFFER_BIT);
    }
}