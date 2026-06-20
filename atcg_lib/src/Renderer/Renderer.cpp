#include <Renderer/Renderer.h>

#include <Core/Assert.h>
#include <Core/Path.h>
#include <Core/SystemRegistry.h>

#include <Renderer/ShaderManager.h>
#include <Renderer/GraphicsAPI.h>
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

    int total_units = GraphicsCommand::getTotalTextureUnits();
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
                                      {TextureFormat::RGBA},    // Color
                                      {TextureFormat::RINT},    // Entity ids
                                      stencil,                  // Stencil mask
                                      {TextureFormat::DEPTH}    // Depth
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
    ATCG_ASSERT(context->isCurrent(), "Context of Renderer not current.");


    const auto& shader = shader_manager->getShader("circle");

    GraphicsCommand::bindVertexArray(quad_vao);

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
    GraphicsCommand::bindPipeline(pipeline);
    if(ibo)
    {
        GraphicsCommand::drawIndexed(static_cast<uint32_t>(ibo->getCount()));
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
    impl->shader_manager->addShaderFromName("volume_hom");
    impl->shader_manager->addShaderFromName("volume_het");
    impl->shader_manager->addShaderFromName("depth_pass_simple");
    impl->shader_manager->addShaderFromName("blit");
    impl->shader_manager->addShaderFromName("mesh_preview");
}

void RendererSystem::use()
{
    impl->context->makeCurrent();
}

void RendererSystem::finishFrame()
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");
#ifndef ATCG_HEADLESS
    auto shader = impl->shader_manager->getShader("screen");
    GraphicsPipeline pipeline =
        GraphicsPipeline().setPrimitiveTopology(PrimitiveTopology::ATCG_TRIANGLES).setShader(shader);

    GraphicsCommand::beginRenderPass(nullptr);
    GraphicsCommand::setViewport(0, 0, impl->screen_fbo->width(), impl->screen_fbo->height());
    GraphicsCommand::bindVertexArray(impl->quad_vao);

    GraphicsCommand::clear();

    shader->setInt("screen_texture", 0);
    shader->selectSubroutine("_getEntityID", "getDefaultID");

    GraphicsCommand::bindTexture(0, impl->screen_fbo->getColorAttachement());
    GraphicsCommand::bindPipeline(pipeline);

    const atcg::ref_ptr<IndexBuffer> ibo = impl->quad_vao->getIndexBuffer();
    GraphicsCommand::drawIndexed(static_cast<uint32_t>(ibo->getCount()));
    GraphicsCommand::endRenderPass();
#endif
    ++impl->frame_counter;
}

void RendererSystem::finish() const
{
    GraphicsCommand::finish();
}

void RendererSystem::resize(const uint32_t& width, const uint32_t& height)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    impl->initFramebuffer(width, height);
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
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    GraphicsCommand::bindVertexArray(vao);

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

    GraphicsCommand::bindPipeline(pipeline);
    if(ibo)
    {
        GraphicsCommand::drawIndexedInstanced(static_cast<uint32_t>(ibo->getCount()), instances);
    }
    else
    {
        GraphicsCommand::drawInstanced(static_cast<uint32_t>(size), instances);
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
    GraphicsCommand::bindVertexArray(impl->quad_vao);
    shader->setInt("screen_texture", 0);

    if(entity_ids)
    {
        shader->setInt("entity_ids", 1);
        GraphicsCommand::bindTexture(1, entity_ids);
        shader->selectSubroutine("_getEntityID", "getFromTextureID");
    }
    else
    {
        shader->selectSubroutine("_getEntityID", "getDefaultID");
    }

    GraphicsCommand::bindTexture(0, img);
    GraphicsCommand::bindPipeline(pipeline);

    const atcg::ref_ptr<IndexBuffer> ibo = impl->quad_vao->getIndexBuffer();
    GraphicsCommand::drawIndexed(static_cast<uint32_t>(ibo->getCount()));
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
            GraphicsCommand::bindStorageBuffer(0, points);

            GraphicsCommand::bindPipeline(pipeline);
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

    pipeline.rasterizer_state.setLineSize(2.0f);

    auto points = impl->cross->getVerticesBuffer();
    GraphicsCommand::bindStorageBuffer(0, points);

    GraphicsCommand::bindPipeline(pipeline);
    drawVAO(impl->cross->getEdgesArray(), camera, glm::mat4(1), pipeline, impl->cross->n_edges());

    shader->setFloat("fall_off_edge", 1000.0f);
}

atcg::ref_ptr<Framebuffer> RendererSystem::getFramebuffer() const
{
    return impl->screen_fbo;
}

atcg::ref_ptr<ShaderManagerSystem> RendererSystem::getShaderManager() const
{
    return impl->shader_manager;
}
}    // namespace atcg