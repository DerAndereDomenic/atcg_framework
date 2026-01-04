#include <Renderer/Renderer.h>
#include <glad/glad.h>

#include <Core/Assert.h>
#include <Core/Path.h>
#include <Core/SystemRegistry.h>

#include <Renderer/ShaderManager.h>
#include <Renderer/RenderAPI.h>
#include <Scene/Components.h>

#include <Scene/Scene.h>
#include <Scene/Entity.h>

#include <queue>

namespace atcg
{
class RendererSystem::Impl
{
public:
    Impl(uint32_t width, uint32_t height, const atcg::ref_ptr<Context>& context);

    ~Impl() = default;

    atcg::ref_ptr<Context> context;
    atcg::ref_ptr<ShaderManagerSystem> shader_manager;

    atcg::ref_ptr<VertexArray> quad_vao;
    atcg::ref_ptr<VertexBuffer> quad_vbo;
    atcg::ref_ptr<IndexBuffer> quad_ibo;

    void initGrid();
    atcg::ref_ptr<Graph> grid;

    void initCross();
    atcg::ref_ptr<Graph> cross;

    void initCube();
    atcg::ref_ptr<Graph> cube;

    void initFramebuffer(uint32_t width, uint32_t height);

    atcg::ref_ptr<Framebuffer> screen_fbo;

    glm::vec4 clear_color;

    uint32_t frame_counter = 0;

    void drawCircle(const glm::vec3& position,
                    const float& radius,
                    const float& thickness,
                    const glm::vec3& color,
                    const atcg::ref_ptr<Camera>& camera = {},
                    uint32_t entity_id                  = -1);

    std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>> texture_ids;
    void freeTextureUnits();

    RenderAPI render_api;
    bool render_pass_started = false;
};

RendererSystem::RendererSystem() {}

RendererSystem::~RendererSystem() {}

RendererSystem::Impl::Impl(uint32_t width, uint32_t height, const atcg::ref_ptr<Context>& context)
{
    this->context = context;

    ATCG_ASSERT(context->isCurrent(), "Context of Renderer not current.");

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

    // Generate CAD grid
    initGrid();
    initCross();

    initCube();

    initFramebuffer(width, height);

    int total_units;
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &total_units);    // TODO
    for(uint32_t i = 0; i < (uint32_t)total_units; ++i)
    {
        texture_ids.push(i);
    }

    ATCG_INFO("RendererSystem supports {0} texture units.", total_units);
}

void RendererSystem::Impl::initGrid()
{
    ATCG_ASSERT(context->isCurrent(), "Context of Renderer not current.");

    int32_t grid_size = 1001;

    std::vector<atcg::Vertex> host_points;
    for(int i = 0; i < grid_size; ++i)
    {
        host_points.push_back(atcg::Vertex(glm::vec3(-(grid_size - 1) / 2 + i, 0.0f, -grid_size / 2), glm::vec3(1)));
        host_points.push_back(atcg::Vertex(glm::vec3(-(grid_size - 1) / 2 + i, 0.0f, grid_size / 2), glm::vec3(1)));

        host_points.push_back(atcg::Vertex(glm::vec3(-grid_size / 2, 0.0f, -(grid_size - 1) / 2 + i), glm::vec3(1)));
        host_points.push_back(atcg::Vertex(glm::vec3(grid_size / 2, 0.0f, -(grid_size - 1) / 2 + i), glm::vec3(1)));
    }

    std::vector<atcg::Edge> edges;

    for(int i = 0; i < 4 * grid_size; i += 2)
    {
        edges.push_back({glm::vec2(i, i + 1), glm::vec3(1), 0.1f});
    }

    grid = atcg::Graph::createGraph(host_points, edges);
}

void RendererSystem::Impl::initCross()
{
    ATCG_ASSERT(context->isCurrent(), "Context of Renderer not current.");

    std::vector<atcg::Vertex> points;
    points.push_back(atcg::Vertex(glm::vec3(-10000.0f, 0.0f, 0.0f), glm::vec3(1)));
    points.push_back(atcg::Vertex(glm::vec3(10000.0f, 0.0f, 0.0f), glm::vec3(1)));

    points.push_back(atcg::Vertex(glm::vec3(0.0f, 0.0f, -10000.0f), glm::vec3(1)));
    points.push_back(atcg::Vertex(glm::vec3(0.0f, 0.0f, 10000.0f), glm::vec3(1)));

    std::vector<atcg::Edge> edges;
    edges.push_back({glm::vec2(0, 1), glm::vec3(1, 0, 0), 0.1f});
    edges.push_back({glm::vec2(2, 3), glm::vec3(0, 0, 1), 0.1f});

    cross = atcg::Graph::createGraph(points, edges);
}

void RendererSystem::Impl::initCube()
{
    ATCG_ASSERT(context->isCurrent(), "Context of Renderer not current.");

    std::vector<atcg::Vertex> points;
    points.push_back(atcg::Vertex(glm::vec3(0.5f, 0.5f, -0.5f), glm::vec3(1)));
    points.push_back(atcg::Vertex(glm::vec3(0.5f, -0.5f, -0.5f), glm::vec3(1)));
    points.push_back(atcg::Vertex(glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(1)));
    points.push_back(atcg::Vertex(glm::vec3(0.5f, -0.5f, 0.5f), glm::vec3(1)));
    points.push_back(atcg::Vertex(glm::vec3(-0.5f, 0.5f, -0.5f), glm::vec3(1)));
    points.push_back(atcg::Vertex(glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(1)));
    points.push_back(atcg::Vertex(glm::vec3(-0.5f, 0.5f, 0.5f), glm::vec3(1)));
    points.push_back(atcg::Vertex(glm::vec3(-0.5f, -0.5f, 0.5f), glm::vec3(1)));

    std::vector<glm::u32vec3> faces;
    faces.push_back(glm::u32vec3(4, 2, 0));
    faces.push_back(glm::u32vec3(2, 7, 3));
    faces.push_back(glm::u32vec3(6, 5, 7));
    faces.push_back(glm::u32vec3(1, 7, 5));
    faces.push_back(glm::u32vec3(0, 3, 1));
    faces.push_back(glm::u32vec3(4, 1, 5));
    faces.push_back(glm::u32vec3(4, 6, 2));
    faces.push_back(glm::u32vec3(2, 6, 7));
    faces.push_back(glm::u32vec3(6, 4, 5));
    faces.push_back(glm::u32vec3(1, 3, 7));
    faces.push_back(glm::u32vec3(0, 2, 3));
    faces.push_back(glm::u32vec3(4, 0, 1));

    cube = atcg::Graph::createTriangleMesh(points, faces);
}

void RendererSystem::Impl::initFramebuffer(uint32_t width, uint32_t height)
{
    ATCG_ASSERT(context->isCurrent(), "Context of Renderer not current.");

    TextureSpecification stencil;
    stencil.format              = TextureFormat::RINT8;
    stencil.sampler.filter_mode = TextureFilterMode::NEAREST;

    FramebufferSpecification spec(width,
                                  height,
                                  1,
                                  {
                                      {TextureFormat::RGBA},          // Color
                                      {TextureFormat::RINT},          // Entity ids
                                      stencil,                        // Stencil mask
                                      {TextureFormat::DEPTH, true}    // Depth
                                  });

    screen_fbo = Framebuffer::create(spec);
}

void RendererSystem::Impl::drawCircle(const glm::vec3& position,
                                      const float& radius,
                                      const float& thickness,
                                      const glm::vec3& color,
                                      const atcg::ref_ptr<Camera>& camera,
                                      uint32_t entity_id)
{
    ATCG_ASSERT(render_pass_started, "Render pass not started in Renderer.");
    ATCG_ASSERT(context->isCurrent(), "Context of Renderer not current.");


    const auto& shader = shader_manager->getShader("circle");

    render_api.bindVertexArray(quad_vao);

    shader->setVec3("flat_color", color);
    shader->setFloat("radius", radius);
    shader->setFloat("thickness", thickness);
    shader->setVec3("position", position);
    shader->setInt("entityID", entity_id);
    if(camera)
    {
        shader->setMVP(glm::mat4(1), camera->getView(), camera->getProjection());
    }

    const atcg::ref_ptr<IndexBuffer> ibo = quad_vao->getIndexBuffer();

    GraphicsPipeline pipeline =
        GraphicsPipeline().setPrimitiveTopology(PrimitiveTopology::ATCG_TRIANGLES).setShader(shader).setShader(shader);
    render_api.bindPipeline(pipeline);
    if(ibo)
    {
        render_api.drawIndexed(static_cast<uint32_t>(ibo->getCount()));
    }
    else
    {
        ATCG_ERROR("Missing IndexBuffer!");
    }
}

void RendererSystem::init(uint32_t width,
                          uint32_t height,
                          const atcg::ref_ptr<Context>& context,
                          const atcg::ref_ptr<ShaderManagerSystem>& shader_manager)
{
    context->makeCurrent();

    ATCG_INFO("OpenGL Renderer:");
    ATCG_INFO("    Vendor: {0}", (const char*)glGetString(GL_VENDOR));
    ATCG_INFO("    Renderer: {0}", (const char*)glGetString(GL_RENDERER));
    ATCG_INFO("    Version: {0}", (const char*)glGetString(GL_VERSION));
    ATCG_INFO("---------------------------------");

    impl = atcg::make_scope<Impl>(width, height, context);

    impl->shader_manager = shader_manager;
    impl->shader_manager->addShaderFromName("base");
    impl->shader_manager->addShaderFromName("flat");
    impl->shader_manager->addShaderFromName("instanced");
    impl->shader_manager->addShaderFromName("edge");
    impl->shader_manager->addShaderFromName("circle");
    impl->shader_manager->addShaderFromName("screen");
    impl->shader_manager->addShaderFromName("cylinder_edge");
    impl->shader_manager->addShaderFromName("equirectangularToCubemap");
    impl->shader_manager->addShaderFromName("skybox");
    impl->shader_manager->addShaderFromName("cubeMapConvolution");
    impl->shader_manager->addShaderFromName("prefilter_cubemap");
    impl->shader_manager->addShaderFromName("vrScreen");
    impl->shader_manager->addShaderFromName("depth_pass");
    impl->shader_manager->addShaderFromName("image_display");
    impl->shader_manager->addShaderFromName("emissive");
    impl->shader_manager->addShaderFromName("tonemap");
}

void RendererSystem::use()
{
    impl->context->makeCurrent();
}

void RendererSystem::beginRenderPass(const atcg::ref_ptr<Framebuffer>& target)
{
    ATCG_ASSERT(!impl->render_pass_started, "Render pass already started.");
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");
    impl->render_api.beginRenderPass(target);
    impl->render_pass_started = true;
}

void RendererSystem::endRenderPass()
{
    ATCG_ASSERT(impl->render_pass_started, "Render pass not started.");
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");
    impl->render_api.endRenderPass();
    impl->render_pass_started = false;
}

void RendererSystem::clear()
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");
    GraphicsPipeline pipeline =
        GraphicsPipeline().setRasterizerState(RasterizerState().setDepthState(DepthState().enableDepthWrite(true)));
    impl->render_api.bindPipeline(pipeline);
    impl->render_api.clear();
}

void RendererSystem::bindTexture(uint32_t slot, const atcg::ref_ptr<Texture>& texture)
{
    impl->render_api.bindTexture(slot, texture);
}

void RendererSystem::finishFrame()
{
    ATCG_ASSERT(!impl->render_pass_started, "Cannot finish frame while render pass is active.");
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");
#ifndef ATCG_HEADLESS
    auto shader = impl->shader_manager->getShader("screen");
    GraphicsPipeline pipeline =
        GraphicsPipeline().setPrimitiveTopology(PrimitiveTopology::ATCG_TRIANGLES).setShader(shader);

    impl->render_api.beginRenderPass(nullptr);
    impl->render_api.setViewport(0, 0, impl->screen_fbo->width(), impl->screen_fbo->height());
    impl->render_api.bindVertexArray(impl->quad_vao);

    impl->render_api.clear();

    shader->setInt("screen_texture", 0);
    shader->selectSubroutine("_getEntityID", "getDefaultID");

    impl->render_api.bindTexture(0, impl->screen_fbo->getColorAttachement());
    impl->render_api.bindPipeline(pipeline);

    const atcg::ref_ptr<IndexBuffer> ibo = impl->quad_vao->getIndexBuffer();
    impl->render_api.drawIndexed(static_cast<uint32_t>(ibo->getCount()));
    impl->render_api.endRenderPass();
#endif
    ++impl->frame_counter;

    GLint maxTextureUnits;
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &maxTextureUnits);    // TODO

    for(int i = 0; i < maxTextureUnits; ++i)
    {
        // TODO
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindTexture(GL_TEXTURE_3D, 0);
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
        glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, 0);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
    }
    glActiveTexture(GL_TEXTURE0);
}

void RendererSystem::finish() const
{
    glFinish();
}

void RendererSystem::resize(const uint32_t& width, const uint32_t& height)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    impl->initFramebuffer(width, height);
}

void RendererSystem::useScreenBuffer() const
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");
    impl->screen_fbo->use();
}

uint32_t RendererSystem::getFrameCounter() const
{
    return impl->frame_counter;
}

uint32_t RendererSystem::popTextureID()
{
    uint32_t id = impl->texture_ids.top();
    impl->texture_ids.pop();
    return id;
}

void RendererSystem::pushTextureID(const uint32_t id)
{
    impl->texture_ids.push(id);
}

void RendererSystem::drawVAO(const atcg::ref_ptr<VertexArray>& vao,
                             const atcg::ref_ptr<Camera>& camera,
                             const glm::mat4& model,
                             const GraphicsPipeline& pipeline,
                             const size_t size,
                             const size_t instances)
{
    ATCG_ASSERT(impl->render_pass_started, "Render pass not started in Renderer.");
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    impl->render_api.bindVertexArray(vao);

    pipeline.shader->setInt("instanced", static_cast<int>(instances > 1));
    if(camera)
    {
        pipeline.shader->setVec3("camera_pos", camera->getPosition());
        pipeline.shader->setVec3("camera_dir", camera->getDirection());
        pipeline.shader->setMVP(model, camera->getView(), camera->getProjection());
    }
    else
    {
        pipeline.shader->setMVP(model);
    }

    const atcg::ref_ptr<IndexBuffer> ibo = vao->getIndexBuffer();

    impl->render_api.bindPipeline(pipeline);
    if(ibo)
    {
        impl->render_api.drawIndexedInstanced(static_cast<uint32_t>(ibo->getCount()), instances);
    }
    else
    {
        impl->render_api.drawInstanced(static_cast<uint32_t>(size), instances);
    }
}

void RendererSystem::drawCircle(const glm::vec3& position,
                                const float& radius,
                                const float& thickness,
                                const glm::vec3& color,
                                const atcg::ref_ptr<Camera>& camera)
{
    impl->drawCircle(position, radius, thickness, color, camera);
}

void RendererSystem::drawImage(const atcg::ref_ptr<Framebuffer>& img)
{
    auto color                          = std::static_pointer_cast<atcg::Texture2D>(img->getColorAttachement(0));
    atcg::ref_ptr<Texture2D> entity_ids = nullptr;
    if(img->getNumberAttachements() > 1)
    {
        auto ids = std::static_pointer_cast<atcg::Texture2D>(img->getColorAttachement(1));
        if(ids->getSpecification().format == atcg::TextureFormat::RINT)
        {
            entity_ids = ids;
        }
    }
    drawImage(color, entity_ids);
}

void RendererSystem::drawImage(const atcg::ref_ptr<Texture2D>& img, const atcg::ref_ptr<Texture2D>& entity_ids)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    auto shader               = impl->shader_manager->getShader("screen");
    GraphicsPipeline pipeline = GraphicsPipeline();

    impl->render_api.beginRenderPass(Framebuffer::currentFramebuffer());
    impl->render_api.bindVertexArray(impl->quad_vao);
    shader->setInt("screen_texture", 0);

    if(entity_ids)
    {
        shader->setInt("entity_ids", 1);
        impl->render_api.bindTexture(1, entity_ids);
        shader->selectSubroutine("_getEntityID", "getFromTextureID");
    }
    else
    {
        shader->selectSubroutine("_getEntityID", "getDefaultID");
    }

    impl->render_api.bindTexture(0, img);
    impl->render_api.bindPipeline(pipeline);

    const atcg::ref_ptr<IndexBuffer> ibo = impl->quad_vao->getIndexBuffer();
    impl->render_api.drawIndexed(static_cast<uint32_t>(ibo->getCount()));
    impl->render_api.endRenderPass();
}

void RendererSystem::drawCADGrid(const atcg::ref_ptr<Camera>& camera, const float& transparency_)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    float distance = glm::abs(camera->getPosition().y);

    auto& shader = impl->shader_manager->getShader("edge");
    GraphicsPipeline pipeline =
        GraphicsPipeline()
            .setPrimitiveTopology(PrimitiveTopology::ATCG_POINTS)
            .setRasterizerState(
                RasterizerState()
                    .enableCulling(false)
                    .setDepthState(DepthState().setDepthFunction(DepthFunction::ATCG_LEQUAL).enableDepthWrite(false))
                    .setLineSize(1.0f))
            .setShader(shader);

    shader->setInt("entityID", -1);
    shader->setFloat("fall_off_edge", distance);
    shader->setVec3("flat_color", glm::vec3(1));

    float edge1 = 1, edge2 = 15;

    float base_transparency = transparency_;
    float resolution        = 0.1f;

    float edges_start[] = {std::numeric_limits<float>::min(),
                           edge1 - 1.0f,
                           edge1 - 2.0f,
                           edge2 - 10.0f,
                           edge2,
                           edge2 - 17.0f,
                           std::numeric_limits<float>::max()};
    float edges_end[]   = {std::numeric_limits<float>::min(),
                           edge1 + 1.0f,
                           edge1,
                           edge2 + 10.0f,
                           edge2 + 3.0f,
                           std::numeric_limits<float>::max()};

    for(int i = 0; i < 3; ++i)
    {
        float transparency = glm::smoothstep(edges_start[2 * i], edges_end[2 * i], distance) -
                             glm::smoothstep(edges_start[2 * i + 1], edges_end[2 * i + 1], distance);

        if(transparency > 0.0f)
        {
            glm::vec3 center = camera->getPosition();
            int32_t x        = static_cast<int32_t>(floor(center.x / resolution + 0.5f));
            int32_t z        = static_cast<int32_t>(floor(center.z / resolution + 0.5f));

            shader->setFloat("base_transparency", base_transparency * transparency);

            auto points = impl->grid->getVerticesBuffer();
            points->bindStorage(0);

            impl->render_api.bindPipeline(pipeline);
            drawVAO(impl->grid->getEdgesArray(),
                    camera,
                    glm::translate(resolution * glm::vec3(x, 0, z)) * glm::scale(glm::vec3(resolution)),
                    pipeline,
                    impl->grid->n_edges());
        }

        resolution *= 10.0f;
    }

    // Reset shader for normal rendering
    shader->setFloat("base_transparency", 1.0f);

    pipeline.render_state.setLineSize(2.0f);

    auto points = impl->cross->getVerticesBuffer();
    points->bindStorage(0);

    impl->render_api.bindPipeline(pipeline);
    drawVAO(impl->cross->getEdgesArray(), camera, glm::mat4(1), pipeline, impl->cross->n_edges());

    shader->setFloat("fall_off_edge", 1000.0f);
}

atcg::ref_ptr<Framebuffer> RendererSystem::getFramebuffer() const
{
    return impl->screen_fbo;
}


torch::Tensor RendererSystem::getFrame(const torch::DeviceType& device) const
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    return impl->screen_fbo->getColorAttachement(0)->getData(device);
}

torch::Tensor RendererSystem::getZBuffer(const torch::DeviceType& device) const
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    auto frame           = impl->screen_fbo->getDepthAttachement();
    uint32_t width       = frame->width();
    uint32_t height      = frame->height();
    torch::Tensor buffer = torch::empty({height, width, 1}, atcg::TensorOptions::floatHostOptions());

    impl->screen_fbo->use();

    glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, buffer.data_ptr());

    return buffer.to(device);
}

int RendererSystem::getEntityIndex(const glm::vec2& mouse) const
{
    // TODO
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    impl->screen_fbo->use();
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    int pixelData;
    glReadPixels((int)mouse.x, (int)mouse.y, 1, 1, GL_RED_INTEGER, GL_INT, &pixelData);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    return pixelData;
}

void RendererSystem::screenshot(const atcg::ref_ptr<Scene>& scene,
                                const atcg::ref_ptr<Camera>& camera,
                                const uint32_t width,
                                const std::string& path)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    auto data = screenshot(scene, camera, width);

    Image img(data);

    img.store(path);
}

void RendererSystem::screenshot(const atcg::ref_ptr<Scene>& scene,
                                const atcg::ref_ptr<Camera>& camera,
                                const uint32_t width,
                                const uint32_t height,
                                const std::string& path)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    atcg::ref_ptr<Framebuffer> screenshot_buffer = atcg::make_ref<Framebuffer>((int)width, (int)height);
    screenshot_buffer->attachColor();
    screenshot_buffer->attachDepth();
    screenshot_buffer->complete();

    atcg::Dictionary context;
    context.setValue("camera", camera);
    context.setValue("target", screenshot_buffer);
    scene->draw(context);    // Starts a renderpass

    auto data = screenshot_buffer->getColorAttachement(0)->getData(atcg::CPU);

    Image img(data);

    img.store(path);

    useScreenBuffer();
}

torch::Tensor
RendererSystem::screenshot(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera, const uint32_t width)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    float height                                 = (float)width / camera->getIntrinsics().aspectRatio();
    atcg::ref_ptr<Framebuffer> screenshot_buffer = atcg::make_ref<Framebuffer>((int)width, (int)height);
    screenshot_buffer->attachColor();
    screenshot_buffer->attachDepth();
    screenshot_buffer->complete();

    screenshot_buffer->use();
    atcg::Dictionary context;
    context.setValue("camera", camera);
    context.setValue("target", screenshot_buffer);
    scene->draw(context);
    useScreenBuffer();

    auto data = screenshot_buffer->getColorAttachement(0)->getData(atcg::CPU);

    return data;
}

atcg::ref_ptr<ShaderManagerSystem> RendererSystem::getShaderManager() const
{
    return impl->shader_manager;
}
}    // namespace atcg