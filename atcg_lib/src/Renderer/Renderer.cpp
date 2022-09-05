#include <Renderer/Renderer.h>
#include <glad/glad.h>
#include <iostream>

#include <Renderer/ShaderManager.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

namespace atcg
{
    Renderer* Renderer::s_renderer = new Renderer;

    class Renderer::Impl
    {
    public:
        Impl();

        ~Impl() = default;

        std::shared_ptr<VertexArray> quad_vao;
        std::shared_ptr<VertexBuffer> quad_vbo;
        std::shared_ptr<IndexBuffer> quad_ibo;
    };

    Renderer::Renderer() {}

    Renderer::~Renderer() {}

    Renderer::Impl::Impl()
    {
        quad_vao = std::make_shared<VertexArray>();

        float vertices[] =
        {
            -1, -1, 0,
            1, -1, 0,
            -1, 1, 0,
            1, 1, 0
        };

        quad_vbo = std::make_shared<VertexBuffer>(vertices, sizeof(vertices));
        quad_vbo->setLayout({
            {ShaderDataType::Float3, "aPosition"}
        });

        quad_vao->addVertexBuffer(quad_vbo);

        uint32_t indices[] = 
        {
            0, 1, 2,
            1, 3, 2
        };

        quad_ibo = std::make_shared<IndexBuffer>(indices, 6);
        quad_vao->setIndexBuffer(quad_ibo);
    }

    void Renderer::init()
    {
        if(!gladLoadGL())
        {
            std::cerr << "Error loading glad!\n";
        }

        s_renderer->impl = std::make_unique<Impl>();

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  
    }

    void Renderer::setClearColor(const glm::vec4& color)
    {
        glClearColor(color.r, color.g, color.b, color.a);
    }

    void Renderer::setViewport(const uint32_t& x, const uint32_t& y, const uint32_t& width, const uint32_t& height)
    {
        glViewport(x, y, width, height);
    }

    void Renderer::clear()
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    void Renderer::draw(const std::shared_ptr<VertexArray>& vao, 
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

    void Renderer::draw(const std::shared_ptr<RenderMesh>& mesh, 
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

    void Renderer::drawPoints(const std::shared_ptr<VertexArray>& vao, 
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

        glPointSize(8);

        if(ibo)
            glDrawElements(GL_POINTS, ibo->getCount(), GL_UNSIGNED_INT, (void*)0);
        else
            std::cerr << "Missing IndexBuffer!\n";
    }

    void Renderer::drawPoints(const std::shared_ptr<RenderMesh>& mesh, 
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

    void Renderer::drawLines(const std::shared_ptr<VertexArray>& vao, 
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

        if(ibo)
            glDrawElements(GL_LINE_STRIP, ibo->getCount(), GL_UNSIGNED_INT, (void*)0);
        else
            std::cerr << "Missing IndexBuffer!\n";
    }

    void Renderer::drawLines(const std::shared_ptr<RenderMesh>& mesh, 
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

        if(ibo)
            glDrawElements(GL_TRIANGLES, ibo->getCount(), GL_UNSIGNED_INT, (void*)0);
        else
            std::cerr << "Missing IndexBuffer!\n";
    }

    void Renderer::drawCircle(const glm::vec3& position,
                              const float& radius,
                              const glm::vec3& color,
                              const std::shared_ptr<Camera>& camera)
    {
        s_renderer->impl->quad_vao->use();
        const auto& shader = ShaderManager::getShader("circle");
        shader->use();
        shader->setVec3("flat_color", color);
        glm::mat4 model = glm::translate(position) * glm::scale(glm::vec3(radius));
        if(camera)
        {
            shader->setMVP(model, camera->getView(), camera->getProjection());
        }
        else
        {
            shader->setMVP(model);
        }

        const std::shared_ptr<IndexBuffer> ibo = s_renderer->impl->quad_vao->getIndexBuffer();

        if(ibo)
            glDrawElements(GL_TRIANGLES, ibo->getCount(), GL_UNSIGNED_INT, (void*)0);
        else
            std::cerr << "Missing IndexBuffer!\n";
    }
}