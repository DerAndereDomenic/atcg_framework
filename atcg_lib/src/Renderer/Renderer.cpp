#include <Renderer/Renderer.h>
#include <glad/glad.h>
#include <iostream>

#include <Renderer/ShaderManager.h>

namespace atcg
{
    Renderer* Renderer::s_renderer = new Renderer;

    void Renderer::initImpl()
    {
        if(!gladLoadGL())
        {
            std::cerr << "Error loading glad!\n";
        }

        glEnable(GL_DEPTH_TEST);
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
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    void Renderer::drawImpl(const std::shared_ptr<VertexArray>& vao, 
                            const std::shared_ptr<Shader>& shader, 
                            const std::shared_ptr<Camera>& camera)
    {
        vao->use();
        shader->use();
        if(camera)
        {
            shader->setVec3("camera_pos", camera->getPosition());
            shader->setVec3("camera_dir", glm::normalize(camera->getPosition() - camera->getLookAt()));
            shader->setMVP(glm::mat4(1), camera->getView(), camera->getProjection());
        }
        else
        {
            shader->setMVP();
        }

        const std::shared_ptr<IndexBuffer> ibo = vao->getIndexBuffer();

        if(ibo)
            glDrawElements(GL_TRIANGLES, ibo->getCount(), GL_UNSIGNED_INT, (void*)0);
        else
            std::cerr << "Missing IndexBuffer!\n";
    }

    void Renderer::drawImpl(const std::shared_ptr<RenderMesh>& mesh, 
                            const std::shared_ptr<Shader>& shader, 
                            const std::shared_ptr<Camera>& camera)
    {
        std::shared_ptr<VertexArray> vao = mesh->getVertexArray();
        vao->use();
        shader->use();
        if(camera)
        {
            shader->setVec3("camera_pos", camera->getPosition());
            shader->setVec3("camera_dir", glm::normalize(camera->getPosition() - camera->getLookAt()));
            shader->setMVP(mesh->getModel(), camera->getView(), camera->getProjection());
        }
        else
        {
            shader->setMVP();
        }

        const std::shared_ptr<IndexBuffer> ibo = vao->getIndexBuffer();

        if(ibo)
            glDrawElements(GL_TRIANGLES, ibo->getCount(), GL_UNSIGNED_INT, (void*)0);
        else
            std::cerr << "Missing IndexBuffer!\n";
    }

    void Renderer::drawPointsImpl(const std::shared_ptr<VertexArray>& vao, 
                                  const glm::vec3& color, 
                                  const std::shared_ptr<Shader>& shader, 
                                  const std::shared_ptr<Camera>& camera)
    {
        vao->use();
        shader->use();
        shader->setVec3("flat_color", color);
        if(camera)
        {
            shader->setMVP(glm::mat4(1), camera->getView(), camera->getProjection());
        }
        else
        {
            shader->setMVP();
        }

        const std::shared_ptr<IndexBuffer> ibo = vao->getIndexBuffer();

        glPointSize(4);

        if(ibo)
            glDrawElements(GL_POINTS, ibo->getCount(), GL_UNSIGNED_INT, (void*)0);
        else
            std::cerr << "Missing IndexBuffer!\n";
    }

    void Renderer::drawPointsImpl(const std::shared_ptr<RenderMesh>& mesh, 
                                  const glm::vec3& color, 
                                  const std::shared_ptr<Shader>& shader, 
                                  const std::shared_ptr<Camera>& camera)
    {
        std::shared_ptr<VertexArray> vao = mesh->getVertexArray();
        vao->use();
        shader->use();
        shader->setVec3("flat_color", color);
        if(camera)
        {
            shader->setMVP(mesh->getModel(), camera->getView(), camera->getProjection());
        }
        else
        {
            shader->setMVP();
        }

        const std::shared_ptr<IndexBuffer> ibo = vao->getIndexBuffer();

        glPointSize(8);

        if(ibo)
            glDrawElements(GL_POINTS, ibo->getCount(), GL_UNSIGNED_INT, (void*)0);
        else
            std::cerr << "Missing IndexBuffer!\n";
    }

    void Renderer::drawLinesImpl(const std::shared_ptr<VertexArray>& vao, 
                                 const glm::vec3& color, 
                                 const std::shared_ptr<Shader>& shader, 
                                 const std::shared_ptr<Camera>& camera)
    {
        vao->use();
        shader->use();
        shader->setVec3("flat_color", color);
        if(camera)
        {
            shader->setMVP(glm::mat4(1), camera->getView(), camera->getProjection());
        }
        else
        {
            shader->setMVP();
        }

        const std::shared_ptr<IndexBuffer> ibo = vao->getIndexBuffer();

        glPointSize(4);

        if(ibo)
            glDrawElements(GL_LINE_STRIP, ibo->getCount(), GL_UNSIGNED_INT, (void*)0);
        else
            std::cerr << "Missing IndexBuffer!\n";
    }

    void Renderer::drawLinesImpl(const std::shared_ptr<RenderMesh>& mesh, 
                                 const glm::vec3& color, 
                                 const std::shared_ptr<Camera>& camera)
    {
        std::shared_ptr<VertexArray> vao = mesh->getVertexArray();
        vao->use();
        const auto& shader = ShaderManager::getShader("edge");
        shader->use();
        shader->setVec3("flat_color", color);
        if(camera)
        {
            shader->setMVP(mesh->getModel(), camera->getView(), camera->getProjection());
        }
        else
        {
            shader->setMVP();
        }

        const std::shared_ptr<IndexBuffer> ibo = vao->getIndexBuffer();

        glPointSize(4);

        if(ibo)
            glDrawElements(GL_TRIANGLES, ibo->getCount(), GL_UNSIGNED_INT, (void*)0);
        else
            std::cerr << "Missing IndexBuffer!\n";
    }
}