#include <Renderer/Renderer.h>
#include <glad/glad.h>

#include <Core/Assert.h>
#include <Core/Path.h>
#include <Core/SystemRegistry.h>

#include <Renderer/ShaderManager.h>
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
    RendererSystem* renderer = nullptr;

    atcg::ref_ptr<VertexArray> quad_vao;
    atcg::ref_ptr<VertexBuffer> quad_vbo;
    atcg::ref_ptr<IndexBuffer> quad_ibo;

    void initGrid();
    atcg::ref_ptr<Graph> grid;

    void initCross();
    atcg::ref_ptr<Graph> cross;

    void initCube();
    atcg::ref_ptr<Graph> cube;

    void initCameraFrustrum();
    atcg::ref_ptr<Graph> camera_frustrum;

    void initFramebuffer(uint32_t num_frames, uint32_t width, uint32_t height);

    Material standard_material;

    atcg::ref_ptr<Texture2D> skybox_texture;
    atcg::ref_ptr<TextureCube> skybox_cubemap;
    atcg::ref_ptr<TextureCube> irradiance_cubemap;
    atcg::ref_ptr<TextureCube> prefiltered_cubemap;
    atcg::ref_ptr<Texture2D> lut;
    bool has_skybox = false;

    atcg::ref_ptr<Framebuffer> screen_fbo;
    atcg::ref_ptr<Framebuffer> screen_fbo_msaa;
    uint32_t msaa_num_samples = 8;
    bool msaa_enabled         = true;

    atcg::ref_ptr<Graph> sphere_mesh;
    atcg::ref_ptr<Graph> cylinder_mesh;
    bool sphere_has_instance   = false;
    bool cylinder_has_instance = false;
    bool culling_enabled       = false;
    CullMode cull_mode;

    uint32_t clear_flag = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT;
    glm::vec4 clear_color;

    float point_size = 1.0f;
    float line_size  = 1.0f;

    uint32_t frame_counter = 0;

    void drawVAO(const atcg::ref_ptr<VertexArray>& vao,
                 const atcg::ref_ptr<Camera>& camera,
                 const glm::vec3& color,
                 const atcg::ref_ptr<Shader>& shader,
                 const glm::mat4& model,
                 GLenum mode,
                 uint32_t size,
                 uint32_t instances = 1);

    void drawPointCloudSpheres(const atcg::ref_ptr<VertexBuffer>& vbo,
                               const atcg::ref_ptr<Camera>& camera,
                               const glm::mat4& model,
                               const glm::vec3& color,
                               const atcg::ref_ptr<Shader>& shader,
                               uint32_t n_instances);

    void drawGrid(const atcg::ref_ptr<VertexBuffer>& points,
                  const atcg::ref_ptr<VertexBuffer>& indices,
                  const atcg::ref_ptr<Shader>& shader,
                  const atcg::ref_ptr<Camera>& camera = {},
                  const glm::mat4& model              = glm::mat4(1),
                  const glm::vec3& color              = glm::vec3(1));

    void drawCircle(const glm::vec3& position,
                    const float& radius,
                    const float& thickness,
                    const glm::vec3& color,
                    const atcg::ref_ptr<Camera>& camera = {},
                    uint32_t entity_id                  = -1);

    template<typename T>
    void drawComponent(Entity entity,
                       const atcg::ref_ptr<Camera>& camera,
                       Scene* scene,
                       const GeometryComponent& geometry,
                       const TransformComponent& transform,
                       const uint32_t entity_id,
                       const atcg::ref_ptr<atcg::Shader>& shader);

    void draw(const atcg::ref_ptr<Graph>& mesh,
              const Material& material,
              uint32_t entity_id,
              const atcg::ref_ptr<Camera>& camera,
              const glm::mat4& model,
              const glm::vec3& color,
              const atcg::ref_ptr<Shader>& shader,
              DrawMode draw_mode);

    void setMaterial(const Material& material, const atcg::ref_ptr<Shader>& shader);
    void setLights(Scene* scene, const atcg::ref_ptr<Shader>& shader);
    void updateShadowmaps(const atcg::ref_ptr<atcg::Scene>& scene, const atcg::ref_ptr<atcg::Camera>& camera);
    atcg::ref_ptr<atcg::Framebuffer> point_light_framebuffer;
    atcg::ref_ptr<atcg::TextureCubeArray> point_light_depth_maps;

    std::vector<uint32_t> used_texture_units;
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

    // Load a sphere
    sphere_mesh = atcg::IO::read_mesh((atcg::resource_directory() / "sphere_low.obj").string());

    cylinder_mesh = atcg::IO::read_mesh((atcg::resource_directory() / "cylinder.obj").string());

    // Generate CAD grid
    initGrid();
    initCross();

    initCube();

    initCameraFrustrum();

    TextureSpecification spec_skybox;
    spec_skybox.width               = 1024;
    spec_skybox.height              = 1024;
    spec_skybox.format              = TextureFormat::RGBAFLOAT;
    spec_skybox.sampler.wrap_mode   = TextureWrapMode::CLAMP_TO_EDGE;
    spec_skybox.sampler.filter_mode = TextureFilterMode::MIPMAP_LINEAR;
    skybox_cubemap                  = atcg::TextureCube::create(spec_skybox);

    TextureSpecification spec_irradiance_cubemap;
    spec_irradiance_cubemap.width             = 32;
    spec_irradiance_cubemap.height            = 32;
    spec_irradiance_cubemap.format            = TextureFormat::RGBAFLOAT;
    spec_irradiance_cubemap.sampler.wrap_mode = TextureWrapMode::CLAMP_TO_EDGE;
    irradiance_cubemap                        = atcg::TextureCube::create(spec_irradiance_cubemap);

    TextureSpecification spec_prefiltered_cubemap;
    spec_prefiltered_cubemap.width               = 128;
    spec_prefiltered_cubemap.height              = 128;
    spec_prefiltered_cubemap.format              = TextureFormat::RGBAFLOAT;
    spec_prefiltered_cubemap.sampler.wrap_mode   = TextureWrapMode::CLAMP_TO_EDGE;
    spec_prefiltered_cubemap.sampler.filter_mode = TextureFilterMode::MIPMAP_LINEAR;
    spec_prefiltered_cubemap.sampler.mip_map     = true;
    prefiltered_cubemap                          = atcg::TextureCube::create(spec_prefiltered_cubemap);

    auto img = IO::imread((atcg::resource_directory() / "LUT.hdr").string());
    TextureSpecification spec_lut;
    spec_lut.width             = img->width();
    spec_lut.height            = img->height();
    spec_lut.format            = TextureFormat::RGBFLOAT;
    spec_lut.sampler.wrap_mode = TextureWrapMode::CLAMP_TO_EDGE;
    lut                        = atcg::Texture2D::create(img, spec_lut);

    initFramebuffer(msaa_num_samples, width, height);

    int total_units;
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &total_units);
    for(uint32_t i = 0; i < (uint32_t)total_units; ++i)
    {
        texture_ids.push(i);
    }

    point_light_framebuffer = atcg::make_ref<atcg::Framebuffer>(1024, 1024);

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

void RendererSystem::Impl::initCameraFrustrum()
{
    ATCG_ASSERT(context->isCurrent(), "Context of Renderer not current.");

    glm::vec3 eye = glm::vec3(0);

    std::vector<atcg::Vertex> points;
    points.push_back(atcg::Vertex(eye, glm::vec3(1)));
    points.push_back(atcg::Vertex(eye + glm::vec3(-0.5, -0.5, 1.0f), glm::vec3(1)));
    points.push_back(atcg::Vertex(eye + glm::vec3(0.5, -0.5, 1.0f), glm::vec3(1)));
    points.push_back(atcg::Vertex(eye + glm::vec3(0.5, 0.5, 1.0f), glm::vec3(1)));
    points.push_back(atcg::Vertex(eye + glm::vec3(-0.5, 0.5, 1.0f), glm::vec3(1)));

    std::vector<atcg::Edge> edges;
    edges.push_back({glm::vec2(0, 1), glm::vec3(1), 0.01f});
    edges.push_back({glm::vec2(0, 2), glm::vec3(1), 0.01f});
    edges.push_back({glm::vec2(0, 3), glm::vec3(1), 0.01f});
    edges.push_back({glm::vec2(0, 4), glm::vec3(1), 0.01f});

    edges.push_back({glm::vec2(1, 2), glm::vec3(1), 0.01f});
    edges.push_back({glm::vec2(2, 3), glm::vec3(1), 0.01f});
    edges.push_back({glm::vec2(3, 4), glm::vec3(1), 0.01f});
    edges.push_back({glm::vec2(4, 1), glm::vec3(1), 0.01f});

    camera_frustrum = atcg::Graph::createGraph(points, edges);
}

void RendererSystem::Impl::initFramebuffer(uint32_t num_samples, uint32_t width, uint32_t height)
{
    ATCG_ASSERT(context->isCurrent(), "Context of Renderer not current.");

    screen_fbo = atcg::make_ref<Framebuffer>(width, height);
    screen_fbo->attachColor();
    TextureSpecification spec_int;
    spec_int.width  = width;
    spec_int.height = height;
    spec_int.format = TextureFormat::RINT;
    screen_fbo->attachTexture(Texture2D::create(spec_int));
    screen_fbo->attachDepth();
    screen_fbo->complete();

    screen_fbo_msaa = atcg::make_ref<Framebuffer>(width, height);
    screen_fbo_msaa->attachColorMultiSample(num_samples);
    screen_fbo_msaa->attachTexture(Texture2DMultiSample::create(num_samples, spec_int));
    screen_fbo_msaa->attachDepthMultiSample(num_samples);
    screen_fbo_msaa->complete();
}

void RendererSystem::Impl::setMaterial(const Material& material, const atcg::ref_ptr<Shader>& shader)
{
    ATCG_ASSERT(context->isCurrent(), "Context of Renderer not current.");

    uint32_t diffuse_id = renderer->popTextureID();
    material.getDiffuseTexture()->use(diffuse_id);
    shader->setInt("texture_diffuse", diffuse_id);
    used_texture_units.push_back(diffuse_id);

    uint32_t normal_id = renderer->popTextureID();
    material.getNormalTexture()->use(normal_id);
    shader->setInt("texture_normal", normal_id);
    used_texture_units.push_back(normal_id);

    uint32_t roughness_id = renderer->popTextureID();
    material.getRoughnessTexture()->use(roughness_id);
    shader->setInt("texture_roughness", roughness_id);
    used_texture_units.push_back(roughness_id);

    uint32_t metallic_id = renderer->popTextureID();
    material.getMetallicTexture()->use(metallic_id);
    shader->setInt("texture_metallic", metallic_id);
    used_texture_units.push_back(metallic_id);

    uint32_t irradiance_id = renderer->popTextureID();
    irradiance_cubemap->use(irradiance_id);
    shader->setInt("irradiance_map", irradiance_id);
    used_texture_units.push_back(irradiance_id);

    uint32_t prefiltered_id = renderer->popTextureID();
    prefiltered_cubemap->use(prefiltered_id);
    shader->setInt("prefilter_map", prefiltered_id);
    used_texture_units.push_back(prefiltered_id);

    uint32_t lut_id = renderer->popTextureID();
    lut->use(lut_id);
    shader->setInt("lut", lut_id);
    used_texture_units.push_back(lut_id);

    shader->setInt("use_ibl", has_skybox);
}

void RendererSystem::Impl::setLights(Scene* scene, const atcg::ref_ptr<Shader>& shader)
{
    auto light_view = scene->getAllEntitiesWith<atcg::PointLightComponent, atcg::TransformComponent>();

    uint32_t num_lights = 0;
    for(auto e: light_view)
    {
        std::stringstream light_index;
        light_index << "[" << num_lights << "]";
        std::string light_index_str = light_index.str();

        atcg::Entity light_entity(e, scene);

        auto& point_light     = light_entity.getComponent<atcg::PointLightComponent>();
        auto& light_transform = light_entity.getComponent<atcg::TransformComponent>();

        shader->setVec3("light_colors" + light_index_str, point_light.color);
        shader->setFloat("light_intensities" + light_index_str, point_light.intensity);
        shader->setVec3("light_positions" + light_index_str, light_transform.getPosition());

        ++num_lights;
    }

    shader->setInt("num_lights", num_lights);
    if(point_light_depth_maps)
    {
        uint32_t shadow_map_id = renderer->popTextureID();
        shader->setInt("shadow_maps", shadow_map_id);
        point_light_depth_maps->use(shadow_map_id);

        used_texture_units.push_back(shadow_map_id);
    }
    else
    {
        ATCG_ASSERT(num_lights == 0, "Shadow map is not initialized but lights are present");
    }
}

void RendererSystem::Impl::updateShadowmaps(const atcg::ref_ptr<atcg::Scene>& scene,
                                            const atcg::ref_ptr<atcg::Camera>& camera)
{
    float n              = 0.1f;
    float f              = 100.0f;
    glm::mat4 projection = glm::perspective(glm::radians(90.0f), 1.0f, n, f);

    const atcg::ref_ptr<Shader>& depth_pass_shader = shader_manager->getShader("depth_pass");
    depth_pass_shader->setFloat("far_plane", f);

    uint32_t active_fbo = atcg::Framebuffer::currentFramebuffer();

    GLint old_viewport[4];
    glGetIntegerv(GL_VIEWPORT, old_viewport);

    auto light_view = scene->getAllEntitiesWith<PointLightComponent, TransformComponent>();

    uint32_t num_lights = 0;
    for(auto e: light_view)
    {
        ++num_lights;
    }

    if(num_lights == 0)
    {
        point_light_depth_maps = nullptr;
        return;
    }

    if(!point_light_depth_maps || point_light_depth_maps->depth() != num_lights)
    {
        atcg::TextureSpecification spec;
        spec.depth             = num_lights;
        spec.width             = 1024;
        spec.height            = 1024;
        spec.format            = atcg::TextureFormat::DEPTH;
        point_light_depth_maps = atcg::TextureCubeArray::create(spec);

        point_light_framebuffer->attachDepth(point_light_depth_maps);
        point_light_framebuffer->complete();
    }

    point_light_framebuffer->use();
    renderer->setViewport(0, 0, point_light_framebuffer->width(), point_light_framebuffer->height());
    renderer->clear();

    uint32_t light_idx = 0;
    for(auto e: light_view)
    {
        atcg::Entity entity(e, scene.get());

        auto& point_light = entity.getComponent<PointLightComponent>();
        auto& transform   = entity.getComponent<TransformComponent>();

        if(!point_light.cast_shadow)
        {
            ++light_idx;
            continue;
        }

        glm::vec3 lightPos = transform.getPosition();
        depth_pass_shader->setVec3("lightPos", lightPos);
        depth_pass_shader->setMat4(
            "shadowMatrices[0]",
            projection * glm::lookAt(lightPos, lightPos + glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
        depth_pass_shader->setMat4(
            "shadowMatrices[1]",
            projection * glm::lookAt(lightPos, lightPos + glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
        depth_pass_shader->setMat4(
            "shadowMatrices[2]",
            projection * glm::lookAt(lightPos, lightPos + glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0)));
        depth_pass_shader->setMat4(
            "shadowMatrices[3]",
            projection * glm::lookAt(lightPos, lightPos + glm::vec3(0.0, -1.0, 0.0), glm::vec3(0.0, 0.0, -1.0)));
        depth_pass_shader->setMat4(
            "shadowMatrices[4]",
            projection * glm::lookAt(lightPos, lightPos + glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, -1.0, 0.0)));
        depth_pass_shader->setMat4(
            "shadowMatrices[5]",
            projection * glm::lookAt(lightPos, lightPos + glm::vec3(0.0, 0.0, -1.0), glm::vec3(0.0, -1.0, 0.0)));
        depth_pass_shader->setInt("light_idx", light_idx);

        const auto& view =
            scene->getAllEntitiesWith<atcg::TransformComponent, atcg::GeometryComponent, atcg::MeshRenderComponent>();

        // Draw scene
        for(auto e: view)
        {
            atcg::Entity entity(e, scene.get());

            auto& transform = entity.getComponent<atcg::TransformComponent>();
            auto& geometry  = entity.getComponent<atcg::GeometryComponent>();

            renderer->draw(geometry.graph, camera, transform.getModel(), glm::vec3(1), depth_pass_shader);
        }

        ++light_idx;
    }

    renderer->setViewport(old_viewport[0], old_viewport[1], old_viewport[2], old_viewport[3]);
    atcg::Framebuffer::bindByID(active_fbo);
}

void RendererSystem::Impl::freeTextureUnits()
{
    for(uint32_t texture_id: used_texture_units)
    {
        renderer->pushTextureID(texture_id);
    }
    used_texture_units.clear();
}

void RendererSystem::Impl::drawVAO(const atcg::ref_ptr<VertexArray>& vao,
                                   const atcg::ref_ptr<Camera>& camera,
                                   const glm::vec3& color,
                                   const atcg::ref_ptr<Shader>& shader,
                                   const glm::mat4& model,
                                   GLenum mode,
                                   uint32_t size,
                                   uint32_t instances)
{
    ATCG_ASSERT(context->isCurrent(), "Context of Renderer not current.");

    vao->use();
    shader->setVec3("flat_color", color);
    shader->setInt("instanced", static_cast<int>(instances > 1));
    if(camera)
    {
        shader->setVec3("camera_pos", camera->getPosition());
        shader->setVec3("camera_dir", camera->getDirection());
        shader->setMVP(model, camera->getView(), camera->getProjection());
    }
    else
    {
        shader->setMVP(model);
    }
    shader->use();

    const atcg::ref_ptr<IndexBuffer> ibo = vao->getIndexBuffer();

    if(ibo)
        glDrawElementsInstanced(mode, static_cast<GLsizei>(ibo->getCount()), GL_UNSIGNED_INT, (void*)0, instances);
    else
        glDrawArraysInstanced(mode, 0, static_cast<GLsizei>(size), instances);
}

void RendererSystem::Impl::drawPointCloudSpheres(const atcg::ref_ptr<VertexBuffer>& vbo,
                                                 const atcg::ref_ptr<Camera>& camera,
                                                 const glm::mat4& model,
                                                 const glm::vec3& color,
                                                 const atcg::ref_ptr<Shader>& shader,
                                                 uint32_t n_instances)
{
    ATCG_ASSERT(context->isCurrent(), "Context of Renderer not current.");

    atcg::ref_ptr<VertexArray> vao_sphere = sphere_mesh->getVerticesArray();
    if(vao_sphere->peekVertexBuffer() != vbo)
    {
        if(sphere_has_instance)
        {
            vao_sphere->popVertexBuffer();
        }
        vao_sphere->pushInstanceBuffer(vbo);
        sphere_has_instance = true;
    }
    glm::mat4 model_new = model;
    shader->setFloat("point_size", point_size);
    drawVAO(vao_sphere, camera, color, shader, model_new, GL_TRIANGLES, sphere_mesh->n_vertices(), n_instances);
}

void RendererSystem::Impl::drawCircle(const glm::vec3& position,
                                      const float& radius,
                                      const float& thickness,
                                      const glm::vec3& color,
                                      const atcg::ref_ptr<Camera>& camera,
                                      uint32_t entity_id)
{
    ATCG_ASSERT(context->isCurrent(), "Context of Renderer not current.");

    quad_vao->use();
    const auto& shader = shader_manager->getShader("circle");
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

    shader->use();
    if(ibo)
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(ibo->getCount()), GL_UNSIGNED_INT, (void*)0);
    else
        ATCG_ERROR("Missing IndexBuffer!");
}

template<>
void RendererSystem::Impl::drawComponent<MeshRenderComponent>(Entity entity,
                                                              const atcg::ref_ptr<Camera>& camera,
                                                              Scene* scene,
                                                              const GeometryComponent& geometry,
                                                              const TransformComponent& transform,
                                                              const uint32_t entity_id,
                                                              const atcg::ref_ptr<atcg::Shader>& override_shader)
{
    MeshRenderComponent renderer = entity.getComponent<MeshRenderComponent>();

    auto shader = override_shader ? override_shader : renderer.shader;

    if(renderer.visible)
    {
        setLights(scene, shader);
        shader->setInt("receive_shadow", (int)renderer.receive_shadow);
        draw(geometry.graph,
             renderer.material,
             entity.entity_handle(),
             camera,
             transform.getModel(),
             glm::vec3(1),
             shader,
             atcg::DrawMode::ATCG_DRAW_MODE_TRIANGLE);
    }
}

template<>
void RendererSystem::Impl::drawComponent<PointRenderComponent>(Entity entity,
                                                               const atcg::ref_ptr<Camera>& camera,
                                                               Scene* scene,
                                                               const GeometryComponent& geometry,
                                                               const TransformComponent& transform,
                                                               const uint32_t entity_id,
                                                               const atcg::ref_ptr<atcg::Shader>& override_shader)
{
    PointRenderComponent renderer = entity.getComponent<PointRenderComponent>();

    auto shader = override_shader ? override_shader : renderer.shader;

    if(renderer.visible)
    {
        setLights(scene, shader);
        this->renderer->setPointSize(renderer.point_size);
        draw(geometry.graph,
             standard_material,
             entity.entity_handle(),
             camera,
             transform.getModel(),
             renderer.color,
             shader,
             atcg::DrawMode::ATCG_DRAW_MODE_POINTS);
    }
}

template<>
void RendererSystem::Impl::drawComponent<PointSphereRenderComponent>(Entity entity,
                                                                     const atcg::ref_ptr<Camera>& camera,
                                                                     Scene* scene,
                                                                     const GeometryComponent& geometry,
                                                                     const TransformComponent& transform,
                                                                     const uint32_t entity_id,
                                                                     const atcg::ref_ptr<atcg::Shader>& override_shader)
{
    PointSphereRenderComponent renderer = entity.getComponent<PointSphereRenderComponent>();

    auto shader = override_shader ? override_shader : renderer.shader;

    if(renderer.visible)
    {
        setLights(scene, shader);
        this->renderer->setPointSize(renderer.point_size);
        draw(geometry.graph,
             renderer.material,
             entity.entity_handle(),
             camera,
             transform.getModel(),
             glm::vec3(1),
             shader,
             atcg::DrawMode::ATCG_DRAW_MODE_POINTS_SPHERE);
    }
}

template<>
void RendererSystem::Impl::drawComponent<EdgeRenderComponent>(Entity entity,
                                                              const atcg::ref_ptr<Camera>& camera,
                                                              Scene* scene,
                                                              const GeometryComponent& geometry,
                                                              const TransformComponent& transform,
                                                              const uint32_t entity_id,
                                                              const atcg::ref_ptr<atcg::Shader>& override_shader)
{
    EdgeRenderComponent renderer = entity.getComponent<EdgeRenderComponent>();

    auto shader = override_shader ? override_shader : shader_manager->getShader("edge");

    if(renderer.visible)
    {
        setLights(scene, shader);
        draw(geometry.graph,
             standard_material,
             entity.entity_handle(),
             camera,
             transform.getModel(),
             renderer.color,
             shader,
             atcg::DrawMode::ATCG_DRAW_MODE_EDGES);
    }
}

template<>
void RendererSystem::Impl::drawComponent<EdgeCylinderRenderComponent>(
    Entity entity,
    const atcg::ref_ptr<Camera>& camera,
    Scene* scene,
    const GeometryComponent& geometry,
    const TransformComponent& transform,
    const uint32_t entity_id,
    const atcg::ref_ptr<atcg::Shader>& override_shader)
{
    EdgeCylinderRenderComponent renderer = entity.getComponent<EdgeCylinderRenderComponent>();

    auto shader = override_shader ? override_shader : shader_manager->getShader("cylinder_edge");

    if(renderer.visible)
    {
        setLights(scene, shader);
        shader->setFloat("edge_radius", renderer.radius);
        draw(geometry.graph,
             renderer.material,
             entity.entity_handle(),
             camera,
             transform.getModel(),
             glm::vec3(1),
             shader,
             atcg::DrawMode::ATCG_DRAW_MODE_EDGES_CYLINDER);
    }
}

template<>
void RendererSystem::Impl::drawComponent<InstanceRenderComponent>(Entity entity,
                                                                  const atcg::ref_ptr<Camera>& camera,
                                                                  Scene* scene,
                                                                  const GeometryComponent& geometry,
                                                                  const TransformComponent& transform,
                                                                  const uint32_t entity_id,
                                                                  const atcg::ref_ptr<atcg::Shader>& override_shader)
{
    InstanceRenderComponent renderer = entity.getComponent<InstanceRenderComponent>();

    auto shader = override_shader ? override_shader : shader_manager->getShader("instanced");

    if(renderer.visible)
    {
        if(geometry.graph->getVerticesArray()->peekVertexBuffer() != renderer.instance_vbo)
        {
            geometry.graph->getVerticesArray()->pushInstanceBuffer(renderer.instance_vbo);
        }

        shader->setInt("entityID", entity_id);
        setMaterial(renderer.material, shader);
        atcg::ref_ptr<VertexArray> vao_mesh      = geometry.graph->getVerticesArray();
        atcg::ref_ptr<VertexBuffer> instance_vbo = vao_mesh->peekVertexBuffer();
        uint32_t n_instances                     = instance_vbo->size() / instance_vbo->getLayout().getStride();
        drawVAO(vao_mesh,
                camera,
                glm::vec3(1),
                shader,
                transform.getModel(),
                GL_TRIANGLES,
                geometry.graph->n_vertices(),
                n_instances);
        freeTextureUnits();
    }
}

void RendererSystem::Impl::draw(const atcg::ref_ptr<Graph>& mesh,
                                const Material& material,
                                uint32_t entity_id,
                                const atcg::ref_ptr<Camera>& camera,
                                const glm::mat4& model,
                                const glm::vec3& color,
                                const atcg::ref_ptr<Shader>& shader,
                                DrawMode draw_mode)
{
    ATCG_ASSERT(context->isCurrent(), "Context of Renderer not current.");

    mesh->unmapAllPointers();
    switch(draw_mode)
    {
        case ATCG_DRAW_MODE_TRIANGLE:
        {
            ATCG_ASSERT(shader, "Tried rendering a mesh without valid shader");
            shader->setInt("entityID", entity_id);
            setMaterial(material, shader);
            drawVAO(mesh->getVerticesArray(),
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
            ATCG_ASSERT(shader, "Tried rendering a point cloud without valid shader");
            shader->setInt("entityID", entity_id);
            setMaterial(material, shader);
            drawVAO(mesh->getVerticesArray(), camera, color, shader, model, GL_POINTS, mesh->n_vertices());
        }
        break;
        case ATCG_DRAW_MODE_POINTS_SPHERE:
        {
            ATCG_ASSERT(shader, "Tried rendering a point cloud without valid shader");
            shader->setInt("entityID", entity_id);
            setMaterial(material, shader);
            drawPointCloudSpheres(mesh->getVerticesArray()->peekVertexBuffer(),
                                  camera,
                                  model,
                                  color,
                                  shader,
                                  mesh->n_vertices());
        }
        break;
        case ATCG_DRAW_MODE_EDGES:
        {
            auto override_shader = shader ? shader : shader_manager->getShader("edge");
            override_shader->setInt("entityID", entity_id);
            setMaterial(material, override_shader);
            atcg::ref_ptr<VertexBuffer> points = mesh->getVerticesBuffer();
            points->bindStorage(0);
            drawVAO(mesh->getEdgesArray(), camera, color, override_shader, model, GL_POINTS, mesh->n_edges(), 1);
        }
        break;
        case ATCG_DRAW_MODE_EDGES_CYLINDER:
        {
            auto override_shader = shader ? shader : shader_manager->getShader("cylinder_edge");
            override_shader->setInt("entityID", entity_id);
            setMaterial(material, override_shader);
            drawGrid(mesh->getVerticesBuffer(), mesh->getEdgesBuffer(), override_shader, camera, model, color);
        }
        break;
        case ATCG_DRAW_MODE_INSTANCED:
        {
            ATCG_ASSERT(shader, "Tried instance rendering without valid shader");
            shader->setInt("entityID", entity_id);
            setMaterial(material, shader);
            atcg::ref_ptr<VertexArray> vao_mesh      = mesh->getVerticesArray();
            atcg::ref_ptr<VertexBuffer> instance_vbo = vao_mesh->peekVertexBuffer();
            uint32_t n_instances                     = instance_vbo->size() / instance_vbo->getLayout().getStride();
            drawVAO(vao_mesh, camera, color, shader, model, GL_TRIANGLES, mesh->n_vertices(), n_instances);
        }
        break;
    }

    freeTextureUnits();
}

void RendererSystem::Impl::drawGrid(const atcg::ref_ptr<VertexBuffer>& points,
                                    const atcg::ref_ptr<VertexBuffer>& indices,
                                    const atcg::ref_ptr<Shader>& shader,
                                    const atcg::ref_ptr<Camera>& camera,
                                    const glm::mat4& model,
                                    const glm::vec3& color)
{
    ATCG_ASSERT(context->isCurrent(), "Context of Renderer not current.");

    atcg::ref_ptr<VertexArray> vao_cylinder = cylinder_mesh->getVerticesArray();
    if(vao_cylinder->peekVertexBuffer() != indices)
    {
        if(cylinder_has_instance)
        {
            vao_cylinder->popVertexBuffer();
        }
        vao_cylinder->pushInstanceBuffer(indices);
        cylinder_has_instance = true;
    }
    uint32_t num_edges = indices->size() / (sizeof(Edge));
    points->bindStorage(0);
    drawVAO(vao_cylinder, camera, color, shader, model, GL_TRIANGLES, cylinder_mesh->n_vertices(), num_edges);
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

    // General settings
    toggleDepthTesting(true);
    toggleCulling(true);
    setCullFace(ATCG_BACK_FACE_CULLING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glEnable(GL_MULTISAMPLE);
    setViewport(0, 0, width, height);

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

    impl->renderer = this;
}

void RendererSystem::use()
{
    impl->context->makeCurrent();
}

void RendererSystem::finishFrame()
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");
#ifndef ATCG_HEADLESS
    Framebuffer::useDefault();
    clear();
    impl->quad_vao->use();
    auto shader = impl->shader_manager->getShader("screen");
    shader->setInt("screen_texture", 0);

    shader->use();
    if(impl->msaa_enabled) impl->screen_fbo->blit(impl->screen_fbo_msaa);
    impl->screen_fbo->getColorAttachement()->use();

    const atcg::ref_ptr<IndexBuffer> ibo = impl->quad_vao->getIndexBuffer();
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(ibo->getCount()), GL_UNSIGNED_INT, (void*)0);
#endif
    ++impl->frame_counter;

    GLint maxTextureUnits;
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &maxTextureUnits);

    for(int i = 0; i < maxTextureUnits; ++i)
    {
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

void RendererSystem::setClearColor(const glm::vec4& color)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");
    impl->clear_color = color;
    glClearColor(color.r, color.g, color.b, color.a);
}

glm::vec4 RendererSystem::getClearColor() const
{
    return impl->clear_color;
}

void RendererSystem::setPointSize(const float& size)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");
    impl->point_size = size;
    glPointSize(size);
}

void RendererSystem::setLineSize(const float& size)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");
    impl->line_size = size;
    glLineWidth(size);
}

void RendererSystem::setMSAA(uint32_t num_samples)
{
    impl->msaa_num_samples = num_samples;
    impl->initFramebuffer(impl->msaa_num_samples, getFramebuffer()->width(), getFramebuffer()->height());
}

uint32_t RendererSystem::getMSAA() const
{
    return impl->msaa_num_samples;
}

void RendererSystem::toggleMSAA(const bool enable)
{
    impl->msaa_enabled = enable;
}

void RendererSystem::toggleDepthTesting(bool enable)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");
    impl->clear_flag = GL_COLOR_BUFFER_BIT;
    switch(enable)
    {
        case true:
        {
            glEnable(GL_DEPTH_TEST);
            impl->clear_flag |= GL_DEPTH_BUFFER_BIT;
        }
        break;
        case false:
        {
            glDisable(GL_DEPTH_TEST);
        }
        break;
    }
}

void RendererSystem::toggleCulling(bool enable)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");
    impl->culling_enabled = enable;
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

void RendererSystem::setCullFace(CullMode mode)
{
    impl->cull_mode = mode;
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");
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


void RendererSystem::setViewport(const uint32_t& x, const uint32_t& y, const uint32_t& width, const uint32_t& height)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");
    glViewport(x, y, width, height);
}

void RendererSystem::setDefaultViewport()
{
    setViewport(0, 0, getFramebuffer()->width(), getFramebuffer()->height());
}

void RendererSystem::setSkybox(const atcg::ref_ptr<Image>& skybox)
{
    setSkybox(atcg::Texture2D::create(skybox));
}

void RendererSystem::setSkybox(const atcg::ref_ptr<Texture2D>& skybox)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    bool culling     = impl->culling_enabled;
    impl->has_skybox = true;
    toggleCulling(false);
    atcg::ref_ptr<PerspectiveCamera> capture_cam = atcg::make_ref<atcg::PerspectiveCamera>();
    glm::mat4 captureProjection                  = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
    glm::mat4 captureViews[]                     = {
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f))};


    capture_cam->setProjection(captureProjection);
    // convert HDR equirectangular environment map to cubemap equivalent

    impl->skybox_texture = skybox;

    uint32_t current_fbo = atcg::Framebuffer::currentFramebuffer();
    int old_viewport[4];
    glGetIntegerv(GL_VIEWPORT, old_viewport);

    uint32_t cubemap_id = popTextureID();

    // * Create a cubemap from the equirectangular map
    {
        atcg::ref_ptr<Shader> equirect_shader = impl->shader_manager->getShader("equirectangularToCubemap");
        float width                           = impl->skybox_cubemap->width();
        float height                          = impl->skybox_cubemap->height();
        Framebuffer captureFBO(width, height);
        captureFBO.attachDepth();

        glViewport(0, 0, width, height);    // don't forget to configure the viewport to the capture dimensions.
        captureFBO.use();

        equirect_shader->use();
        impl->skybox_texture->use(cubemap_id);
        equirect_shader->setInt("equirectangularMap", cubemap_id);
        for(unsigned int i = 0; i < 6; ++i)
        {
            capture_cam->setView(captureViews[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER,
                                   GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                   impl->skybox_cubemap->getID(),
                                   0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            draw(impl->cube, capture_cam, glm::mat4(1), glm::vec3(1), equirect_shader);
            // renderCube();    // renders a 1x1 cube
        }

        impl->skybox_cubemap->generateMipmaps();
    }

    // * Convolution of cube map for irradiance map
    {
        atcg::ref_ptr<Shader> cubeconv_shader = impl->shader_manager->getShader("cubeMapConvolution");
        float width                           = impl->irradiance_cubemap->width();
        float height                          = impl->irradiance_cubemap->height();
        Framebuffer captureFBO(width, height);
        captureFBO.attachDepth();

        glViewport(0, 0, width, height);    // don't forget to configure the viewport to the capture dimensions.
        captureFBO.use();

        cubeconv_shader->use();
        impl->skybox_cubemap->use(cubemap_id);
        cubeconv_shader->setInt("skybox", cubemap_id);
        for(unsigned int i = 0; i < 6; ++i)
        {
            capture_cam->setView(captureViews[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER,
                                   GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                   impl->irradiance_cubemap->getID(),
                                   0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            draw(impl->cube, capture_cam, glm::mat4(1), glm::vec3(1), cubeconv_shader);
            // renderCube();    // renders a 1x1 cube
        }
    }

    // * Prefilter environment map
    {
        atcg::ref_ptr<Shader> prefilter_shader = impl->shader_manager->getShader("prefilter_cubemap");
        float width                            = impl->prefiltered_cubemap->width();
        float height                           = impl->prefiltered_cubemap->height();

        prefilter_shader->use();
        prefilter_shader->setInt("skybox", cubemap_id);
        unsigned int max_mip_levels = 5;
        for(unsigned int mip = 0; mip < max_mip_levels; ++mip)
        {
            unsigned int mip_width  = impl->prefiltered_cubemap->width() * std::pow(0.5, mip);
            unsigned int mip_height = impl->prefiltered_cubemap->height() * std::pow(0.5, mip);

            // Recreate captureFBO with new resolution
            Framebuffer captureFBO(mip_width, mip_height);
            captureFBO.attachDepth();
            captureFBO.use();

            impl->skybox_cubemap->use(cubemap_id);

            glViewport(0, 0, mip_width, mip_height);

            float roughness = (float)mip / (float)(max_mip_levels - 1);
            prefilter_shader->setFloat("roughness", roughness);
            for(unsigned int i = 0; i < 6; ++i)
            {
                capture_cam->setView(captureViews[i]);
                glFramebufferTexture2D(GL_FRAMEBUFFER,
                                       GL_COLOR_ATTACHMENT0,
                                       GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                       impl->prefiltered_cubemap->getID(),
                                       mip);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                draw(impl->cube, capture_cam, glm::mat4(1), glm::vec3(1), prefilter_shader);
                // renderCube();    // renders a 1x1 cube
            }
        }
    }

    pushTextureID(cubemap_id);
    Framebuffer::bindByID(current_fbo);
    setViewport(old_viewport[0], old_viewport[1], old_viewport[2], old_viewport[3]);
    toggleCulling(culling);
}

bool RendererSystem::hasSkybox() const
{
    return impl->has_skybox;
}

void RendererSystem::removeSkybox()
{
    impl->has_skybox = false;
}

atcg::ref_ptr<Texture2D> RendererSystem::getSkyboxTexture() const
{
    return impl->skybox_texture;
}

atcg::ref_ptr<TextureCube> RendererSystem::getSkyboxCubemap() const
{
    return impl->skybox_cubemap;
}

void RendererSystem::resize(const uint32_t& width, const uint32_t& height)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    setViewport(0, 0, width, height);
    impl->initFramebuffer(impl->msaa_num_samples, width, height);
}

void RendererSystem::useScreenBuffer() const
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");
    impl->msaa_enabled ? impl->screen_fbo_msaa->use() : impl->screen_fbo->use();
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

void RendererSystem::clear() const
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");
    glClear(impl->clear_flag);

    if(Framebuffer::currentFramebuffer() == impl->screen_fbo->getID() ||
       Framebuffer::currentFramebuffer() == impl->screen_fbo_msaa->getID())
    {
        int value = -1;
        impl->screen_fbo->getColorAttachement(1)->fill(&value);
        impl->screen_fbo_msaa->getColorAttachement(1)->fill(&value);
    }
}

void RendererSystem::draw(const atcg::ref_ptr<Graph>& mesh,
                          const atcg::ref_ptr<Camera>& camera,
                          const glm::mat4& model,
                          const glm::vec3& color,
                          const atcg::ref_ptr<Shader>& shader,
                          DrawMode draw_mode)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    impl->draw(mesh, impl->standard_material, -1, camera, model, color, shader, draw_mode);
}

template<typename T>
void RendererSystem::drawComponent(Entity entity,
                                   const atcg::ref_ptr<atcg::Camera>& camera,
                                   const atcg::ref_ptr<atcg::Shader>& shader)
{
    if(!entity.hasComponent<T>()) return;

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

    uint32_t entity_id           = entity.entity_handle();
    TransformComponent transform = entity.getComponent<TransformComponent>();
    GeometryComponent geometry   = entity.getComponent<GeometryComponent>();

    if(!geometry.graph)
    {
        ATCG_WARN("Entity does have geometry component but mesh is empty");
        return;
    }

    Scene* scene = entity._scene;

    geometry.graph->unmapAllPointers();

    impl->drawComponent<T>(entity, camera, scene, geometry, transform, entity_id, shader);
}

// drawEntity
void RendererSystem::draw(Entity entity, const atcg::ref_ptr<Camera>& camera)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    if(entity.hasComponent<CustomRenderComponent>())
    {
        CustomRenderComponent renderer = entity.getComponent<CustomRenderComponent>();
        renderer.callback(entity, camera);
    }

    drawComponent<MeshRenderComponent>(entity, camera);
    drawComponent<PointRenderComponent>(entity, camera);
    drawComponent<PointSphereRenderComponent>(entity, camera);
    drawComponent<EdgeRenderComponent>(entity, camera);
    drawComponent<EdgeCylinderRenderComponent>(entity, camera);
    drawComponent<InstanceRenderComponent>(entity, camera);
}

// drawScene
void RendererSystem::draw(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    drawSkybox(camera);

    impl->updateShadowmaps(scene, camera);

    const auto& view = scene->getAllEntitiesWith<atcg::TransformComponent>();

    for(auto e: view)
    {
        Entity entity(e, scene.get());
        RendererSystem::draw(entity, camera);
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
    drawImage(std::static_pointer_cast<atcg::Texture2D>(img->getColorAttachement(0)));
}

void RendererSystem::drawImage(const atcg::ref_ptr<Texture2D>& img)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    impl->quad_vao->use();
    auto shader = impl->shader_manager->getShader("screen");
    shader->setInt("screen_texture", 0);

    shader->use();
    img->use();

    const atcg::ref_ptr<IndexBuffer> ibo = impl->quad_vao->getIndexBuffer();
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(ibo->getCount()), GL_UNSIGNED_INT, (void*)0);
}

void RendererSystem::drawSkybox(const atcg::ref_ptr<Camera>& camera)
{
    if(impl->has_skybox)
    {
        uint32_t skybox_id = popTextureID();
        glDepthMask(GL_FALSE);
        glDepthFunc(GL_LEQUAL);
        bool culling = impl->culling_enabled;
        toggleCulling(false);
        impl->shader_manager->getShader("skybox")->use();
        impl->shader_manager->getShader("skybox")->setInt("skybox", skybox_id);
        impl->skybox_cubemap->use(skybox_id);

        draw(impl->cube, camera, glm::mat4(1), glm::vec3(1), impl->shader_manager->getShader("skybox"));

        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);
        toggleCulling(culling);
        pushTextureID(skybox_id);
    }
}

void RendererSystem::drawCameras(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    const auto& view = scene->getAllEntitiesWith<atcg::CameraComponent>();

    for(auto e: view)
    {
        Entity entity(e, scene.get());
        setLineSize(2.0f);
        uint32_t entity_id = entity.entity_handle();
        impl->shader_manager->getShader("edge")->setInt("entityID", entity_id);
        atcg::CameraComponent& comp          = entity.getComponent<CameraComponent>();
        atcg::ref_ptr<PerspectiveCamera> cam = std::dynamic_pointer_cast<PerspectiveCamera>(comp.camera);
        float aspect_ratio                   = cam->getAspectRatio();
        glm::mat4 scale                      = glm::scale(
            glm::vec3(aspect_ratio, 1.0f, -0.5f / glm::tan(glm::radians(cam->getFOV()) / 2.0f)) * comp.render_scale);
        glm::mat4 model = glm::inverse(cam->getView()) * scale;
        draw(impl->camera_frustrum,
             camera,
             model,
             comp.color,
             impl->shader_manager->getShader("edge"),
             atcg::DrawMode::ATCG_DRAW_MODE_EDGES);


        if(comp.image)
        {
            uint32_t id          = popTextureID();
            bool culling_enabled = impl->culling_enabled;
            toggleCulling(false);
            model = model * glm::translate(glm::vec3(0, 0, 1)) * glm::scale(glm::vec3(0.5));
            impl->shader_manager->getShader("image_display")->setInt("screen_texture", id);
            impl->shader_manager->getShader("image_display")->setInt("entityID", entity_id);
            comp.image->use(id);
            impl->drawVAO(impl->quad_vao,
                          camera,
                          comp.color,
                          impl->shader_manager->getShader("image_display"),
                          model,
                          GL_TRIANGLES,
                          impl->quad_vao->getIndexBuffer()->getCount());
            toggleCulling(culling_enabled);
            pushTextureID(id);
        }
        else if(comp.render_preview && comp.preview)
        {
            uint32_t id          = popTextureID();
            bool culling_enabled = impl->culling_enabled;
            toggleCulling(false);
            model = model * glm::translate(glm::vec3(0, 0, 1)) * glm::scale(glm::vec3(0.5));
            impl->shader_manager->getShader("image_display")->setInt("screen_texture", id);
            impl->shader_manager->getShader("image_display")->setInt("entityID", entity_id);
            comp.preview->getColorAttachement(0)->use(id);
            impl->drawVAO(impl->quad_vao,
                          camera,
                          comp.color,
                          impl->shader_manager->getShader("image_display"),
                          model,
                          GL_TRIANGLES,
                          impl->quad_vao->getIndexBuffer()->getCount());
            toggleCulling(culling_enabled);
            pushTextureID(id);
        }
    }
}

void RendererSystem::drawLights(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    const auto& view = scene->getAllEntitiesWith<atcg::PointLightComponent, atcg::TransformComponent>();
    for(auto e: view)
    {
        Entity entity(e, scene.get());

        auto& transform   = entity.getComponent<atcg::TransformComponent>();
        auto& point_light = entity.getComponent<atcg::PointLightComponent>();

        impl->drawCircle(transform.getPosition(), 0.1f, 1.0f, point_light.color, camera, entity.entity_handle());
    }
}

void RendererSystem::drawCADGrid(const atcg::ref_ptr<Camera>& camera, const float& transparency_)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    float distance     = glm::abs(camera->getPosition().y);
    float current_size = impl->line_size;

    setLineSize(1.0f);

    auto& shader = impl->shader_manager->getShader("edge");
    shader->setInt("entityID", -1);
    shader->setFloat("fall_off_edge", distance);

    float edge1 = 1, edge2 = 15;

    glDepthMask(GL_FALSE);
    glDepthFunc(GL_LEQUAL);

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

            draw(impl->grid,
                 camera,
                 glm::translate(resolution * glm::vec3(x, 0, z)) * glm::scale(glm::vec3(resolution)),
                 glm::vec3(1),
                 shader,
                 atcg::DrawMode::ATCG_DRAW_MODE_EDGES);
        }

        resolution *= 10.0f;
    }

    // Reset shader for normal rendering
    shader->setFloat("base_transparency", 1.0f);

    setLineSize(2.0f);
    draw(impl->cross, camera, glm::mat4(1), glm::vec3(1), shader, atcg::DrawMode::ATCG_DRAW_MODE_EDGES);
    setLineSize(current_size);

    shader->setFloat("fall_off_edge", 1000.0f);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);
}

atcg::ref_ptr<Framebuffer> RendererSystem::getFramebuffer() const
{
    return impl->screen_fbo;
}

atcg::ref_ptr<Framebuffer> RendererSystem::getFramebufferMSAA() const
{
    return impl->screen_fbo_msaa;
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

    atcg::ref_ptr<PerspectiveCamera> cam         = std::dynamic_pointer_cast<PerspectiveCamera>(camera);
    atcg::ref_ptr<Framebuffer> screenshot_buffer = atcg::make_ref<Framebuffer>((int)width, (int)height);
    screenshot_buffer->attachColor();
    screenshot_buffer->attachDepth();
    screenshot_buffer->complete();

    screenshot_buffer->use();
    clear();
    setViewport(0, 0, width, height);
    draw(scene, cam);
    useScreenBuffer();
    setDefaultViewport();

    auto data = screenshot_buffer->getColorAttachement(0)->getData(atcg::CPU);

    Image img(data);

    img.store(path);
}

torch::Tensor
RendererSystem::screenshot(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera, const uint32_t width)
{
    ATCG_ASSERT(impl->context->isCurrent(), "Context of Renderer not current.");

    atcg::ref_ptr<PerspectiveCamera> cam         = std::dynamic_pointer_cast<PerspectiveCamera>(camera);
    float height                                 = (float)width / cam->getAspectRatio();
    atcg::ref_ptr<Framebuffer> screenshot_buffer = atcg::make_ref<Framebuffer>((int)width, (int)height);
    screenshot_buffer->attachColor();
    screenshot_buffer->attachDepth();
    screenshot_buffer->complete();

    screenshot_buffer->use();
    clear();
    setViewport(0, 0, width, height);
    draw(scene, cam);
    useScreenBuffer();
    setDefaultViewport();

    auto data = screenshot_buffer->getColorAttachement(0)->getData(atcg::CPU);

    return data;
}
}    // namespace atcg