#include <Renderer/Renderer.h>
#include <glad/glad.h>
#include <iostream>

#include <Core/Log.h>

#include <Renderer/ShaderManager.h>
#include <Scene/Components.h>

#include <Scene/Scene.h>
#include <Scene/Entity.h>

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

    atcg::ref_ptr<Graph> sphere_mesh;
    atcg::ref_ptr<Graph> cylinder_mesh;
    bool sphere_has_instance   = false;
    bool cylinder_has_instance = false;

    uint32_t clear_flag = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT;

    float point_size = 1.0f;
    float line_size  = 1.0f;

    void drawPointCloudSpheres(const atcg::ref_ptr<VertexBuffer>& vbo,
                               const atcg::ref_ptr<Camera>& camera,
                               const glm::mat4& model,
                               const glm::vec3& color,
                               const atcg::ref_ptr<Shader>& shader,
                               uint32_t n_instances);

    // Render methods
    void drawVAO(const atcg::ref_ptr<VertexArray>& vao,
                 const atcg::ref_ptr<Camera>& camera,
                 const glm::vec3& color,
                 const atcg::ref_ptr<Shader>& shader,
                 const glm::mat4& model,
                 GLenum mode,
                 uint32_t size,
                 uint32_t instances = 1);

    void drawGrid(const atcg::ref_ptr<VertexBuffer>& points,
                  const atcg::ref_ptr<VertexBuffer>& indices,
                  const atcg::ref_ptr<Shader>& shader,
                  const atcg::ref_ptr<Camera>& camera = {},
                  const glm::mat4& model              = glm::mat4(1),
                  const glm::vec3& color              = glm::vec3(1));
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

        quad_vao->pushVertexBuffer(quad_vbo);

        uint32_t indices[] = {0, 1, 2, 1, 3, 2};

        quad_ibo = atcg::make_ref<IndexBuffer>(indices, 6);
        quad_vao->setIndexBuffer(quad_ibo);
    }

    // Generate cube
    initCube();

    // Load a sphere
    sphere_mesh = atcg::IO::read_mesh("res/sphere_low.obj");

    cylinder_mesh = atcg::IO::read_mesh("res/cylinder.obj");

    screen_fbo = atcg::make_ref<Framebuffer>(width, height);
    screen_fbo->attachColor();
    screen_fbo->attachTexture(Texture2D::createIntTexture(width, height));
    screen_fbo->attachDepth();
    screen_fbo->complete();
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

    cube_vao->pushVertexBuffer(cube_vbo);
}

void Renderer::init(uint32_t width, uint32_t height)
{
    ATCG_INFO("OpenGL Renderer:");
    ATCG_INFO("    Vendor: {0}", (const char*)glGetString(GL_VENDOR));
    ATCG_INFO("    Renderer: {0}", (const char*)glGetString(GL_RENDERER));
    ATCG_INFO("    Version: {0}", (const char*)glGetString(GL_VERSION));
    ATCG_INFO("---------------------------------");

    s_renderer->impl = atcg::make_scope<Impl>(width, height);

    // General settings
    toggleDepthTesting(true);
    toggleCulling(true);
    setCullFace(ATCG_BACK_FACE_CULLING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void Renderer::finishFrame()
{
    Framebuffer::useDefault();
    clear();
    s_renderer->impl->quad_vao->use();
    auto shader = ShaderManager::getShader("screen");
    shader->setInt("screen_texture", 0);

    shader->use();
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

void Renderer::setLineSize(const float& size)
{
    s_renderer->impl->line_size = size;
    glLineWidth(size);
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
    s_renderer->impl->screen_fbo->attachTexture(Texture2D::createIntTexture(width, height));
    s_renderer->impl->screen_fbo->attachDepth();
    s_renderer->impl->screen_fbo->complete();
}

void Renderer::useScreenBuffer()
{
    s_renderer->impl->screen_fbo->use();
}

atcg::ref_ptr<Framebuffer> Renderer::getFramebuffer()
{
    return s_renderer->impl->screen_fbo;
}

void Renderer::clear()
{
    glClear(s_renderer->impl->clear_flag);

    if(Framebuffer::currentFramebuffer() == s_renderer->impl->screen_fbo->getID())
    {
        int value = -1;
        glClearTexImage(s_renderer->impl->screen_fbo->getColorAttachement(1)->getID(),
                        0,
                        GL_RED_INTEGER,
                        GL_INT,
                        &value);
    }
}

void Renderer::toggleDepthTesting(bool enable)
{
    s_renderer->impl->clear_flag = GL_COLOR_BUFFER_BIT;
    switch(enable)
    {
        case true:
        {
            glEnable(GL_DEPTH_TEST);
            s_renderer->impl->clear_flag |= GL_DEPTH_BUFFER_BIT;
        }
        break;
        case false:
        {
            glDisable(GL_DEPTH_TEST);
        }
        break;
    }
}

void Renderer::toggleCulling(bool enable)
{
    switch(enable)
    {
        case true:
        {
            glEnable(GL_CULL_FACE);
        }
        break;
        case false:
        {
            glDisable(GL_CULL_FACE);
        }
        break;
    }
}

void Renderer::setCullFace(CullMode mode)
{
    switch(mode)
    {
        case CullMode::ATCG_BACK_FACE_CULLING:
        {
            glCullFace(GL_BACK);
        }
        break;
        case CullMode::ATCG_FRONT_FACE_CULLING:
        {
            glCullFace(GL_FRONT);
        }
        break;
        case CullMode::ATCG_BOTH_FACE_CULLING:
        {
            glCullFace(GL_FRONT_AND_BACK);
        }
        break;
    }
}

void Renderer::draw(const atcg::ref_ptr<VertexArray>& vao,
                    const atcg::ref_ptr<Camera>& camera,
                    const glm::mat4& model,
                    const glm::vec3& color,
                    const atcg::ref_ptr<Shader>& shader,
                    DrawMode draw_mode)
{
    switch(draw_mode)
    {
        case ATCG_DRAW_MODE_TRIANGLE:
        {
            s_renderer->impl->drawVAO(vao, camera, color, shader, model, GL_TRIANGLES, 1e6);    // TODO
        }
        break;
        case ATCG_DRAW_MODE_POINTS:
        {
            s_renderer->impl->drawVAO(vao, camera, color, shader, model, GL_POINTS, 1e6);
        }
        break;
        case ATCG_DRAW_MODE_POINTS_SPHERE:
        {
            throw std::logic_error {"Not implemented"};
        }
        break;
        case ATCG_DRAW_MODE_EDGES:
        {
            s_renderer->impl->drawVAO(vao, camera, color, ShaderManager::getShader("edge"), model, GL_LINE_STRIP, 1e6);
        }
        break;
        case ATCG_DRAW_MODE_EDGES_CYLINDER:
        {
            throw std::logic_error {"Not implemented"};
        }
        break;
    }
}

void Renderer::draw(const atcg::ref_ptr<Graph>& mesh,
                    const atcg::ref_ptr<Camera>& camera,
                    const glm::mat4& model,
                    const glm::vec3& color,
                    const atcg::ref_ptr<Shader>& shader,
                    DrawMode draw_mode)
{
    switch(draw_mode)
    {
        case ATCG_DRAW_MODE_TRIANGLE:
        {
            s_renderer->impl->drawVAO(mesh->getVerticesArray(),
                                      camera,
                                      color,
                                      shader,
                                      model,
                                      GL_TRIANGLES,
                                      mesh->n_vertices());    // TODO
        }
        break;
        case ATCG_DRAW_MODE_POINTS:
        {
            s_renderer->impl
                ->drawVAO(mesh->getVerticesArray(), camera, color, shader, model, GL_POINTS, mesh->n_vertices());
        }
        break;
        case ATCG_DRAW_MODE_POINTS_SPHERE:
        {
            s_renderer->impl->drawPointCloudSpheres(mesh->getVerticesArray()->peekVertexBuffer(),
                                                    camera,
                                                    model,
                                                    color,
                                                    shader,
                                                    mesh->n_vertices());
        }
        break;
        case ATCG_DRAW_MODE_EDGES:
        {
            atcg::ref_ptr<VertexBuffer> points = mesh->getVerticesBuffer();
            points->bindStorage(0);
            s_renderer->impl->drawVAO(mesh->getEdgesArray(),
                                      camera,
                                      color,
                                      ShaderManager::getShader("edge"),
                                      model,
                                      GL_POINTS,
                                      mesh->n_edges(),
                                      1);
        }
        break;
        case ATCG_DRAW_MODE_EDGES_CYLINDER:
        {
            s_renderer->impl->drawGrid(mesh->getVerticesBuffer(),
                                       mesh->getEdgesBuffer(),
                                       ShaderManager::getShader("cylinder_edge"),
                                       camera,
                                       model,
                                       color);
        }
        break;
    }
}

void Renderer::draw(Entity entity, const atcg::ref_ptr<Camera>& camera)
{
    if(!entity.hasComponent<TransformComponent>())
    {
        ATCG_WARN("Entity does not have transform component!");
        return;
    }

    if(!entity.hasComponent<GeometryComponent>())
    {
        ATCG_WARN("Entity does not have geometry component!");
        return;
    }

    uint32_t entity_id           = (uint32_t)entity._entity_handle;
    TransformComponent transform = entity.getComponent<TransformComponent>();
    GeometryComponent geometry   = entity.getComponent<GeometryComponent>();

    // renderer.shader->setInt("entityID", entity_id);

    if(entity.hasComponent<MeshRenderComponent>())
    {
        MeshRenderComponent renderer = entity.getComponent<MeshRenderComponent>();
        renderer.shader->setInt("entityID", entity_id);
        Renderer::draw(geometry.graph,
                       camera,
                       transform.getModel(),
                       renderer.color,
                       renderer.shader,
                       renderer.draw_mode);
    }

    if(entity.hasComponent<PointRenderComponent>())
    {
        PointRenderComponent renderer = entity.getComponent<PointRenderComponent>();
        renderer.shader->setInt("entityID", entity_id);
        Renderer::draw(geometry.graph,
                       camera,
                       transform.getModel(),
                       renderer.color,
                       renderer.shader,
                       renderer.draw_mode);
    }

    if(entity.hasComponent<PointSphereRenderComponent>())
    {
        PointSphereRenderComponent renderer = entity.getComponent<PointSphereRenderComponent>();
        renderer.shader->setInt("entityID", entity_id);
        Renderer::draw(geometry.graph,
                       camera,
                       transform.getModel(),
                       renderer.color,
                       renderer.shader,
                       renderer.draw_mode);
    }

    if(entity.hasComponent<EdgeRenderComponent>())
    {
        EdgeRenderComponent renderer = entity.getComponent<EdgeRenderComponent>();
        ShaderManager::getShader("edge")->setInt("entityID", entity_id);
        Renderer::draw(geometry.graph,
                       camera,
                       transform.getModel(),
                       renderer.color,
                       ShaderManager::getShader("edge"),
                       renderer.draw_mode);
    }

    if(entity.hasComponent<EdgeCylinderRenderComponent>())
    {
        EdgeCylinderRenderComponent renderer = entity.getComponent<EdgeCylinderRenderComponent>();
        ShaderManager::getShader("edge")->setInt("entityID", entity_id);
        Renderer::draw(geometry.graph,
                       camera,
                       transform.getModel(),
                       renderer.color,
                       ShaderManager::getShader("cylinder_edge"),
                       renderer.draw_mode);
    }
}

void Renderer::draw(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera)
{
    const auto& view = scene->getAllEntitiesWith<atcg::TransformComponent>();

    for(auto e: view)
    {
        Entity entity(e, scene.get());
        Renderer::draw(entity, camera);
    }
}

void Renderer::Impl::drawPointCloudSpheres(const atcg::ref_ptr<VertexBuffer>& vbo,
                                           const atcg::ref_ptr<Camera>& camera,
                                           const glm::mat4& model,
                                           const glm::vec3& color,
                                           const atcg::ref_ptr<Shader>& shader,
                                           uint32_t n_instances)
{
    atcg::ref_ptr<VertexArray> vao_sphere = sphere_mesh->getVerticesArray();
    if(vao_sphere->peekVertexBuffer() != vbo)
    {
        if(s_renderer->impl->sphere_has_instance) { vao_sphere->popVertexBuffer(); }
        vao_sphere->pushInstanceBuffer(vbo);
        s_renderer->impl->sphere_has_instance = true;
    }
    glm::mat4 model_new = model;
    shader->setFloat("point_size", s_renderer->impl->point_size);
    s_renderer->impl->drawVAO(vao_sphere,
                              camera,
                              color,
                              shader,
                              model_new,
                              GL_TRIANGLES,
                              s_renderer->impl->sphere_mesh->n_vertices(),
                              n_instances);
}

void Renderer::Impl::drawVAO(const atcg::ref_ptr<VertexArray>& vao,
                             const atcg::ref_ptr<Camera>& camera,
                             const glm::vec3& color,
                             const atcg::ref_ptr<Shader>& shader,
                             const glm::mat4& model,
                             GLenum mode,
                             uint32_t size,
                             uint32_t instances)
{
    vao->use();
    shader->setVec3("flat_color", color);
    shader->setInt("instanced", static_cast<int>(instances > 1));
    if(camera)
    {
        shader->setVec3("camera_pos", camera->getPosition());
        shader->setVec3("camera_dir", camera->getDirection());
        shader->setMVP(model, camera->getView(), camera->getProjection());
    }
    else { shader->setMVP(model); }
    shader->use();

    const atcg::ref_ptr<IndexBuffer> ibo = vao->getIndexBuffer();

    if(ibo)
        glDrawElementsInstanced(mode, static_cast<GLsizei>(ibo->getCount()), GL_UNSIGNED_INT, (void*)0, instances);
    else
        glDrawArraysInstanced(mode, 0, static_cast<GLsizei>(size), instances);
}

void Renderer::drawCircle(const glm::vec3& position,
                          const float& radius,
                          const glm::vec3& color,
                          const atcg::ref_ptr<Camera>& camera)
{
    s_renderer->impl->quad_vao->use();
    const auto& shader = ShaderManager::getShader("circle");
    shader->setVec3("flat_color", color);
    glm::mat4 model = glm::translate(position) * glm::scale(glm::vec3(radius));
    if(camera) { shader->setMVP(model, camera->getView(), camera->getProjection()); }
    else { shader->setMVP(model); }

    const atcg::ref_ptr<IndexBuffer> ibo = s_renderer->impl->quad_vao->getIndexBuffer();

    shader->use();
    if(ibo)
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(ibo->getCount()), GL_UNSIGNED_INT, (void*)0);
    else
        ATCG_ERROR("Missing IndexBuffer!");
}

void Renderer::Impl::drawGrid(const atcg::ref_ptr<VertexBuffer>& points,
                              const atcg::ref_ptr<VertexBuffer>& indices,
                              const atcg::ref_ptr<Shader>& shader,
                              const atcg::ref_ptr<Camera>& camera,
                              const glm::mat4& model,
                              const glm::vec3& color)
{
    atcg::ref_ptr<VertexArray> vao_cylinder = cylinder_mesh->getVerticesArray();
    if(vao_cylinder->peekVertexBuffer() != indices)
    {
        if(cylinder_has_instance) { vao_cylinder->popVertexBuffer(); }
        vao_cylinder->pushInstanceBuffer(indices);
        cylinder_has_instance = true;
    }
    uint32_t num_edges = indices->size() / (sizeof(Edge));
    points->bindStorage(0);
    drawVAO(vao_cylinder, camera, color, shader, model, GL_TRIANGLES, cylinder_mesh->n_vertices(), num_edges);
}

int Renderer::getEntityIndex(const glm::vec2& mouse)
{
    useScreenBuffer();
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    int pixelData;
    glReadPixels((int)mouse.x, (int)mouse.y, 1, 1, GL_RED_INTEGER, GL_INT, &pixelData);
    return pixelData;
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