#include <Renderer/Renderer.h>
#include <glad/glad.h>
#include <iostream>

#include <Core/Log.h>

#include <Renderer/ShaderManager.h>
#include <Renderer/Framebuffer.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

namespace atcg
{
Renderer* Renderer::s_renderer = new Renderer;

class Renderer::Impl
{
public:
    Impl(uint32_t width, uint32_t height);

    ~Impl() = default;

    atcg::ref_ptr<VertexArray> quad_vao;
    atcg::ref_ptr<VertexBuffer> quad_vbo;
    atcg::ref_ptr<IndexBuffer> quad_ibo;

    void initCube();
    atcg::ref_ptr<VertexArray> cube_vao;
    atcg::ref_ptr<VertexBuffer> cube_vbo;

    atcg::ref_ptr<VertexBuffer> grid_vbo;

    atcg::ref_ptr<Framebuffer> screen_fbo;

    float point_size = 8;

    // Render methods
    void drawVAO(const atcg::ref_ptr<VertexArray>& vao,
                 const atcg::ref_ptr<Camera>& camera,
                 const glm::vec3& color,
                 const atcg::ref_ptr<Shader>& shader,
                 const glm::mat4& model,
                 GLenum mode,
                 uint32_t size);

    void drawEdges(const atcg::ref_ptr<VertexArray>& vao, const atcg::ref_ptr<Camera>& camera, const glm::vec3& color);

    void drawEdges(const atcg::ref_ptr<Mesh>& mesh, const atcg::ref_ptr<Camera>& camera, const glm::vec3& color);
};

Renderer::Renderer() {}

Renderer::~Renderer() {}

Renderer::Impl::Impl(uint32_t width, uint32_t height)
{
    // Generate quad
    {
        quad_vao = atcg::make_ref<VertexArray>();

        float vertices[] = {-1, -1, 0, 0, 0, 1, -1, 0, 1, 0, -1, 1, 0, 0, 1, 1, 1, 0, 1, 1};

        quad_vbo = atcg::make_ref<VertexBuffer>(vertices, sizeof(vertices));
        quad_vbo->setLayout({{ShaderDataType::Float3, "aPosition"}, {ShaderDataType::Float2, "aUV"}});

        quad_vao->addVertexBuffer(quad_vbo);

        uint32_t indices[] = {0, 1, 2, 1, 3, 2};

        quad_ibo = atcg::make_ref<IndexBuffer>(indices, 6);
        quad_vao->setIndexBuffer(quad_ibo);
    }

    // Generate cube
    initCube();

    screen_fbo = atcg::make_ref<Framebuffer>(width, height);
    screen_fbo->attachColor();
    screen_fbo->attachDepth();
    screen_fbo->verify();
}

void Renderer::Impl::initCube()
{
    cube_vao = atcg::make_ref<VertexArray>();

    float vertices[] = {
        -0.5f, -0.5f, -0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,  0.5f,  -0.5f,
        0.5f,  0.5f,  -0.5f, -0.5f, 0.5f,  -0.5f, -0.5f, -0.5f, -0.5f,

        -0.5f, -0.5f, 0.5f,  0.5f,  -0.5f, 0.5f,  0.5f,  0.5f,  0.5f,
        0.5f,  0.5f,  0.5f,  -0.5f, 0.5f,  0.5f,  -0.5f, -0.5f, 0.5f,

        -0.5f, 0.5f,  0.5f,  -0.5f, 0.5f,  -0.5f, -0.5f, -0.5f, -0.5f,
        -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, 0.5f,  -0.5f, 0.5f,  0.5f,

        0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  -0.5f, 0.5f,  -0.5f, -0.5f,
        0.5f,  -0.5f, -0.5f, 0.5f,  -0.5f, 0.5f,  0.5f,  0.5f,  0.5f,

        -0.5f, -0.5f, -0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,  -0.5f, 0.5f,
        0.5f,  -0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,  -0.5f, -0.5f, -0.5f,

        -0.5f, 0.5f,  -0.5f, 0.5f,  0.5f,  -0.5f, 0.5f,  0.5f,  0.5f,
        0.5f,  0.5f,  0.5f,  -0.5f, 0.5f,  0.5f,  -0.5f, 0.5f,  -0.5f,
    };

    cube_vbo = atcg::make_ref<VertexBuffer>(vertices, sizeof(vertices));
    cube_vbo->setLayout({{ShaderDataType::Float3, "aPosition"}});

    cube_vao->addVertexBuffer(cube_vbo);
}

void Renderer::init(uint32_t width, uint32_t height)
{
    ATCG_INFO("OpenGL Renderer:");
    ATCG_INFO("    Vendor: {0}", (const char*)glGetString(GL_VENDOR));
    ATCG_INFO("    Renderer: {0}", (const char*)glGetString(GL_RENDERER));
    ATCG_INFO("    Version: {0}", (const char*)glGetString(GL_VERSION));
    ATCG_INFO("---------------------------------");

    s_renderer->impl = atcg::make_scope<Impl>(width, height);
}

void Renderer::finishFrame()
{
    Framebuffer::useDefault();
    clear();
    s_renderer->impl->quad_vao->use();
    auto shader = ShaderManager::getShader("screen");
    shader->use();
    shader->setInt("screen_texture", 0);

    s_renderer->impl->screen_fbo->getColorAttachement()->use();

    const atcg::ref_ptr<IndexBuffer> ibo = s_renderer->impl->quad_vao->getIndexBuffer();
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(ibo->getCount()), GL_UNSIGNED_INT, (void*)0);
}

void Renderer::setClearColor(const glm::vec4& color)
{
    glClearColor(color.r, color.g, color.b, color.a);
}

void Renderer::setPointSize(const float& size)
{
    s_renderer->impl->point_size = size;
    glPointSize(size);
}

void Renderer::setViewport(const uint32_t& x, const uint32_t& y, const uint32_t& width, const uint32_t& height)
{
    glViewport(x, y, width, height);
}

void Renderer::resize(const uint32_t& width, const uint32_t& height)
{
    setViewport(0, 0, width, height);
    s_renderer->impl->screen_fbo = atcg::make_ref<Framebuffer>(width, height);
    s_renderer->impl->screen_fbo->attachColor();
    s_renderer->impl->screen_fbo->attachDepth();
    s_renderer->impl->screen_fbo->verify();
}

void Renderer::useScreenBuffer()
{
    s_renderer->impl->screen_fbo->use();
}

void Renderer::clear()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Renderer::draw(const atcg::ref_ptr<VertexArray>& vao,
                    const atcg::ref_ptr<Camera>& camera,
                    const glm::vec3& color,
                    const atcg::ref_ptr<Shader>& shader,
                    DrawMode draw_mode)
{
    switch(draw_mode)
    {
        case ATCG_DRAW_MODE_TRIANGLE:
        {
            s_renderer->impl->drawVAO(vao, camera, color, shader, glm::mat4(1), GL_TRIANGLES, 1e6);    // TODO
        }
        break;
        case ATCG_DRAW_MODE_POINTS:
        {
            s_renderer->impl->drawVAO(vao, camera, color, shader, glm::mat4(1), GL_POINTS, 1e6);
        }
        break;
        case ATCG_DRAW_MODE_POINTS_SPHERE:
        {
            throw std::logic_error {"Not implemented"};
        }
        break;
        case ATCG_DRAW_MODE_EDGES:
        {
            s_renderer->impl
                ->drawVAO(vao, camera, color, ShaderManager::getShader("edge"), glm::mat4(1), GL_LINE_STRIP, 1e6);
        }
        break;
    }
}

void Renderer::draw(const atcg::ref_ptr<Mesh>& mesh,
                    const atcg::ref_ptr<Camera>& camera,
                    const glm::vec3& color,
                    const atcg::ref_ptr<Shader>& shader,
                    DrawMode draw_mode)
{
    switch(draw_mode)
    {
        case ATCG_DRAW_MODE_TRIANGLE:
        {
            s_renderer->impl->drawVAO(mesh->getVertexArray(),
                                      camera,
                                      color,
                                      shader,
                                      mesh->getModel(),
                                      GL_TRIANGLES,
                                      mesh->n_vertices());    // TODO
        }
        break;
        case ATCG_DRAW_MODE_POINTS:
        {
            s_renderer->impl->drawVAO(mesh->getVertexArray(),
                                      camera,
                                      color,
                                      shader,
                                      mesh->getModel(),
                                      GL_POINTS,
                                      mesh->n_vertices());
        }
        break;
        case ATCG_DRAW_MODE_POINTS_SPHERE:
        {
            throw std::logic_error {"Not implemented"};
        }
        break;
        case ATCG_DRAW_MODE_EDGES:
        {
            s_renderer->impl->drawVAO(mesh->getVertexArray(),
                                      camera,
                                      color,
                                      ShaderManager::getShader("edge"),
                                      mesh->getModel(),
                                      GL_TRIANGLES,
                                      mesh->n_vertices());
        }
        break;
    }
}

void Renderer::draw(const atcg::ref_ptr<PointCloud>& cloud,
                    const atcg::ref_ptr<Camera>& camera,
                    const glm::vec3& color,
                    const atcg::ref_ptr<Shader>& shader,
                    DrawMode draw_mode)
{
    switch(draw_mode)
    {
        case ATCG_DRAW_MODE_TRIANGLE:
        {
            throw std::invalid_argument("PointCloud cannot be rendered as triangle mesh!");
        }
        break;
        case ATCG_DRAW_MODE_POINTS:
        {
            s_renderer->impl
                ->drawVAO(cloud->getVertexArray(), camera, color, shader, glm::mat4(1), GL_POINTS, cloud->n_vertices());
        }
        break;
        case ATCG_DRAW_MODE_POINTS_SPHERE:
        {
            throw std::logic_error {"Not implemented"};
        }
        break;
        case ATCG_DRAW_MODE_EDGES:
        {
            throw std::invalid_argument("PointCloud cannot be rendered as edges!");
        }
        break;
    }
}

void Renderer::Impl::drawVAO(const atcg::ref_ptr<VertexArray>& vao,
                             const atcg::ref_ptr<Camera>& camera,
                             const glm::vec3& color,
                             const atcg::ref_ptr<Shader>& shader,
                             const glm::mat4& model,
                             GLenum mode,
                             uint32_t size)
{
    vao->use();
    shader->use();
    shader->setVec3("flat_color", color);
    if(camera)
    {
        shader->setVec3("camera_pos", camera->getPosition());
        shader->setVec3("camera_dir", camera->getDirection());
        shader->setMVP(model, camera->getView(), camera->getProjection());
    }
    else { shader->setMVP(); }

    const atcg::ref_ptr<IndexBuffer> ibo = vao->getIndexBuffer();

    if(ibo)
        glDrawElements(mode, static_cast<GLsizei>(ibo->getCount()), GL_UNSIGNED_INT, (void*)0);
    else
        glDrawArrays(mode, 0, static_cast<GLsizei>(size));
}

void Renderer::drawCircle(const glm::vec3& position,
                          const float& radius,
                          const glm::vec3& color,
                          const atcg::ref_ptr<Camera>& camera)
{
    s_renderer->impl->quad_vao->use();
    const auto& shader = ShaderManager::getShader("circle");
    shader->use();
    shader->setVec3("flat_color", color);
    glm::mat4 model = glm::translate(position) * glm::scale(glm::vec3(radius));
    if(camera) { shader->setMVP(model, camera->getView(), camera->getProjection()); }
    else { shader->setMVP(model); }

    const atcg::ref_ptr<IndexBuffer> ibo = s_renderer->impl->quad_vao->getIndexBuffer();

    if(ibo)
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(ibo->getCount()), GL_UNSIGNED_INT, (void*)0);
    else
        std::cerr << "Missing IndexBuffer!\n";
}

std::vector<uint8_t> Renderer::getFrame()
{
    auto frame      = s_renderer->impl->screen_fbo->getColorAttachement();
    uint32_t width  = frame->width();
    uint32_t height = frame->height();
    std::vector<uint8_t> buffer(width * height * 4);

    useScreenBuffer();

    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, (void*)buffer.data());

    return buffer;
}

std::vector<float> Renderer::getZBuffer()
{
    auto frame      = s_renderer->impl->screen_fbo->getDepthAttachement();
    uint32_t width  = frame->width();
    uint32_t height = frame->height();
    std::vector<float> buffer(width * height);

    useScreenBuffer();

    glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, (void*)buffer.data());

    return buffer;
}
}    // namespace atcg