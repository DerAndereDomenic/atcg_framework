#include <Renderer/Renderer.h>
#include <glad/glad.h>
#include <iostream>

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

    std::shared_ptr<VertexArray> quad_vao;
    std::shared_ptr<VertexBuffer> quad_vbo;
    std::shared_ptr<IndexBuffer> quad_ibo;

    void initCube();
    std::shared_ptr<VertexArray> cube_vao;
    std::shared_ptr<VertexBuffer> cube_vbo;

    std::shared_ptr<VertexBuffer> grid_vbo;

    std::shared_ptr<Framebuffer> screen_fbo;

    float point_size = 8;
};

Renderer::Renderer() {}

Renderer::~Renderer() {}

Renderer::Impl::Impl(uint32_t width, uint32_t height)
{
    // Generate quad
    {
        quad_vao = std::make_shared<VertexArray>();

        float vertices[] = {-1, -1, 0, 0, 0, 1, -1, 0, 1, 0, -1, 1, 0, 0, 1, 1, 1, 0, 1, 1};

        quad_vbo = std::make_shared<VertexBuffer>(vertices, sizeof(vertices));
        quad_vbo->setLayout({{ShaderDataType::Float3, "aPosition"}, {ShaderDataType::Float2, "aUV"}});

        quad_vao->addVertexBuffer(quad_vbo);

        uint32_t indices[] = {0, 1, 2, 1, 3, 2};

        quad_ibo = std::make_shared<IndexBuffer>(indices, 6);
        quad_vao->setIndexBuffer(quad_ibo);
    }

    // Generate cube
    initCube();

    screen_fbo = std::make_shared<Framebuffer>(width, height);
    screen_fbo->attachColor();
    screen_fbo->attachDepth();
    screen_fbo->verify();
}

void Renderer::Impl::initCube()
{
    cube_vao = std::make_shared<VertexArray>();

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

    cube_vbo = std::make_shared<VertexBuffer>(vertices, sizeof(vertices));
    cube_vbo->setLayout({{ShaderDataType::Float3, "aPosition"}});

    cube_vao->addVertexBuffer(cube_vbo);
}

void Renderer::init(uint32_t width, uint32_t height)
{
    if(!gladLoadGL()) { std::cerr << "Error loading glad!\n"; }

    s_renderer->impl = std::make_unique<Impl>(width, height);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
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

    const std::shared_ptr<IndexBuffer> ibo = s_renderer->impl->quad_vao->getIndexBuffer();
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
    s_renderer->impl->screen_fbo = std::make_shared<Framebuffer>(width, height);
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

void Renderer::draw(const std::shared_ptr<VertexArray>& vao,
                    const std::shared_ptr<Shader>& shader,
                    const std::shared_ptr<Camera>& camera)
{
    vao->use();
    shader->use();
    if(camera)
    {
        shader->setVec3("camera_pos", camera->getPosition());
        shader->setVec3("camera_dir", camera->getDirection());
        shader->setMVP(glm::mat4(1), camera->getView(), camera->getProjection());
    }
    else { shader->setMVP(); }

    const std::shared_ptr<IndexBuffer> ibo = vao->getIndexBuffer();

    if(ibo)
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(ibo->getCount()), GL_UNSIGNED_INT, (void*)0);
    else
        std::cerr << "Missing IndexBuffer!\n";
}

void Renderer::draw(const std::shared_ptr<Mesh>& mesh,
                    const std::shared_ptr<Shader>& shader,
                    const std::shared_ptr<Camera>& camera)
{
    std::shared_ptr<VertexArray> vao = mesh->getVertexArray();
    vao->use();
    shader->use();
    if(camera)
    {
        shader->setVec3("camera_pos", camera->getPosition());
        shader->setVec3("camera_dir", camera->getDirection());
        shader->setMVP(mesh->getModel(), camera->getView(), camera->getProjection());
    }
    else { shader->setMVP(); }

    const std::shared_ptr<IndexBuffer> ibo = vao->getIndexBuffer();

    if(ibo)
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(ibo->getCount()), GL_UNSIGNED_INT, (void*)0);
    else
        std::cerr << "Missing IndexBuffer!\n";
}

void Renderer::draw(const std::shared_ptr<PointCloud>& cloud,
                    const std::shared_ptr<Shader>& shader,
                    const std::shared_ptr<Camera>& camera)
{
    std::shared_ptr<VertexArray> vao = cloud->getVertexArray();
    vao->use();
    shader->use();
    shader->setVec3("flat_color", glm::vec3(1));
    if(camera)
    {
        shader->setVec3("camera_pos", camera->getPosition());
        shader->setVec3("camera_dir", camera->getDirection());
        shader->setMVP(glm::mat4(1), camera->getView(), camera->getProjection());
    }
    else { shader->setMVP(); }

    glPointSize(s_renderer->impl->point_size);

    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(cloud->n_vertices()));
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
        shader->setVec3("camera_pos", camera->getPosition());
        shader->setVec3("camera_dir", camera->getDirection());
        shader->setMVP(glm::mat4(1), camera->getView(), camera->getProjection());
    }
    else { shader->setMVP(); }

    const std::shared_ptr<IndexBuffer> ibo = vao->getIndexBuffer();

    glPointSize(s_renderer->impl->point_size);

    if(ibo)
        glDrawElements(GL_POINTS, static_cast<GLsizei>(ibo->getCount()), GL_UNSIGNED_INT, (void*)0);
    else
        std::cerr << "Missing IndexBuffer!\n";
}

void Renderer::drawPoints(const std::shared_ptr<Mesh>& mesh,
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
        shader->setVec3("camera_pos", camera->getPosition());
        shader->setVec3("camera_dir", camera->getDirection());
        shader->setMVP(mesh->getModel(), camera->getView(), camera->getProjection());
    }
    else { shader->setMVP(); }

    const std::shared_ptr<IndexBuffer> ibo = vao->getIndexBuffer();

    glPointSize(s_renderer->impl->point_size);

    if(ibo)
        glDrawElements(GL_POINTS, static_cast<GLsizei>(ibo->getCount()), GL_UNSIGNED_INT, (void*)0);
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
        shader->setVec3("camera_pos", camera->getPosition());
        shader->setVec3("camera_dir", camera->getDirection());
        shader->setMVP(glm::mat4(1), camera->getView(), camera->getProjection());
    }
    else { shader->setMVP(); }

    const std::shared_ptr<IndexBuffer> ibo = vao->getIndexBuffer();

    if(ibo)
        glDrawElements(GL_LINE_STRIP, static_cast<GLsizei>(ibo->getCount()), GL_UNSIGNED_INT, (void*)0);
    else
        std::cerr << "Missing IndexBuffer!\n";
}

void Renderer::drawLines(const std::shared_ptr<Mesh>& mesh,
                         const glm::vec3& color,
                         const std::shared_ptr<Camera>& camera)
{
    std::shared_ptr<VertexArray> vao = mesh->getVertexArray();
    vao->use();
    const auto& shader = ShaderManager::getShader("edge");
    shader->use();
    shader->setVec3("flat_color", color);
    if(camera) { shader->setMVP(mesh->getModel(), camera->getView(), camera->getProjection()); }
    else { shader->setMVP(); }

    const std::shared_ptr<IndexBuffer> ibo = vao->getIndexBuffer();

    if(ibo)
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(ibo->getCount()), GL_UNSIGNED_INT, (void*)0);
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
    if(camera) { shader->setMVP(model, camera->getView(), camera->getProjection()); }
    else { shader->setMVP(model); }

    const std::shared_ptr<IndexBuffer> ibo = s_renderer->impl->quad_vao->getIndexBuffer();

    if(ibo)
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(ibo->getCount()), GL_UNSIGNED_INT, (void*)0);
    else
        std::cerr << "Missing IndexBuffer!\n";
}

void Renderer::drawGrid(const GridDimension& grid_dimension, const std::shared_ptr<Camera>& camera, bool reset)
{
    s_renderer->impl->cube_vao->use();
    const auto& shader = ShaderManager::getShader("grid");

    Grid<int> dummy(grid_dimension.origin, grid_dimension.num_voxels, grid_dimension.voxel_length, false);
    if(!s_renderer->impl->grid_vbo || reset)
    {
        s_renderer->impl->initCube();
        std::vector<glm::vec3> positions;

        for(uint32_t i = 0; i < dummy.voxels_per_volume(); ++i)
        {
            glm::vec3 pos = dummy.voxel2position(dummy.index2voxel(i));
            positions.push_back(pos);
        }

        s_renderer->impl->grid_vbo =
            std::make_shared<VertexBuffer>(positions.data(), positions.size() * sizeof(glm::vec3));
        s_renderer->impl->grid_vbo->setLayout({{ShaderDataType::Float3, "aPosition"}});

        s_renderer->impl->cube_vao->addInstanceBuffer(s_renderer->impl->grid_vbo);
    }

    shader->use();
    shader->setVec3("flat_color", glm::vec3(0));
    glm::mat4 model = glm::scale(glm::vec3(grid_dimension.voxel_length));
    if(camera) { shader->setMVP(model, camera->getView(), camera->getProjection()); }
    else { shader->setMVP(model); }

    glDrawArraysInstanced(GL_TRIANGLES, 0, 36, dummy.voxels_per_volume());
}
}    // namespace atcg