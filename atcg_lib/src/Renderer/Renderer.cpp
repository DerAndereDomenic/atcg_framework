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

    void Renderer::drawImpl(const std::shared_ptr<VertexArray>& vao, 
                            const std::shared_ptr<Shader>& shader, 
                            const std::shared_ptr<Camera>& camera)
    {
        vao->use();
        shader->use();
        if(camera)
            shader->setMVP(glm::mat4(1), camera->getView(), camera->getProjection());

        const std::shared_ptr<IndexBuffer> ibo = vao->getIndexBuffer();

        if(ibo)
            glDrawElements(GL_TRIANGLES, ibo->getCount(), GL_UNSIGNED_INT, (void*)0);
        else
            std::cerr << "Missing IndexBuffer!\n";
    }
}