#include <Renderer/Renderer.h>
#include <glad/glad.h>

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

    void initGrid();
    atcg::ref_ptr<Graph> grid;

    void initCross();
    atcg::ref_ptr<Graph> cross;

    void initCube();
    atcg::ref_ptr<Graph> cube;

    void initCameraFrustrum();
    atcg::ref_ptr<Graph> camera_frustrum;

    Material standard_material;

    atcg::ref_ptr<Texture2D> skybox_texture;
    atcg::ref_ptr<TextureCube> skybox_cubemap;
    atcg::ref_ptr<TextureCube> irradiance_cubemap;
    atcg::ref_ptr<TextureCube> prefiltered_cubemap;
    atcg::ref_ptr<Texture2D> lut;
    bool has_skybox = false;

    atcg::ref_ptr<Framebuffer> screen_fbo;

    atcg::ref_ptr<Graph> sphere_mesh;
    atcg::ref_ptr<Graph> cylinder_mesh;
    bool sphere_has_instance   = false;
    bool cylinder_has_instance = false;
    bool culling_enabled       = false;

    uint32_t clear_flag = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT;
    glm::vec4 clear_color;

    float point_size = 1.0f;
    float line_size  = 1.0f;

    uint32_t frame_counter = 0;

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

    void setMaterial(const Material& material, const atcg::ref_ptr<Shader>& shader);

    enum TextureBindings
    {
        DIFFUSE_TEXTURE   = 0,
        NORMAL_TEXTURE    = 1,
        ROUGHNESS_TEXTURE = 2,
        METALLIC_TEXTURE  = 3,
        IRRADIANCE_MAP    = 4,
        PREFILTER_MAP     = 5,
        LUT_TEXTURE       = 6,
        SKYBOX_TEXTURE    = 7,
        COUNT
    };
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

    // Load a sphere
    sphere_mesh = atcg::IO::read_mesh("res/sphere_low.obj");

    cylinder_mesh = atcg::IO::read_mesh("res/cylinder.obj");

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

    auto img = IO::imread("res/LUT.hdr");
    TextureSpecification spec_lut;
    spec_lut.width             = img->width();
    spec_lut.height            = img->height();
    spec_lut.format            = TextureFormat::RGBFLOAT;
    spec_lut.sampler.wrap_mode = TextureWrapMode::CLAMP_TO_EDGE;
    lut                        = atcg::Texture2D::create(img, spec_lut);

    screen_fbo = atcg::make_ref<Framebuffer>(width, height);
    screen_fbo->attachColor();
    TextureSpecification spec_int;
    spec_int.width  = width;
    spec_int.height = height;
    spec_int.format = TextureFormat::RINT;
    screen_fbo->attachTexture(Texture2D::create(spec_int));
    screen_fbo->attachDepth();
    screen_fbo->complete();
}

void Renderer::Impl::initGrid()
{
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

void Renderer::Impl::initCross()
{
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

void Renderer::Impl::initCube()
{
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

void Renderer::Impl::initCameraFrustrum()
{
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

void Renderer::Impl::setMaterial(const Material& material, const atcg::ref_ptr<Shader>& shader)
{
    material.getDiffuseTexture()->use(Renderer::Impl::TextureBindings::DIFFUSE_TEXTURE);
    shader->setInt("texture_diffuse", Renderer::Impl::TextureBindings::DIFFUSE_TEXTURE);

    material.getNormalTexture()->use(Renderer::Impl::TextureBindings::NORMAL_TEXTURE);
    shader->setInt("texture_normal", Renderer::Impl::TextureBindings::NORMAL_TEXTURE);

    material.getRoughnessTexture()->use(Renderer::Impl::TextureBindings::ROUGHNESS_TEXTURE);
    shader->setInt("texture_roughness", Renderer::Impl::TextureBindings::ROUGHNESS_TEXTURE);

    material.getMetallicTexture()->use(Renderer::Impl::TextureBindings::METALLIC_TEXTURE);
    shader->setInt("texture_metallic", Renderer::Impl::TextureBindings::METALLIC_TEXTURE);

    irradiance_cubemap->use(Renderer::Impl::TextureBindings::IRRADIANCE_MAP);
    shader->setInt("irradiance_map", Renderer::Impl::TextureBindings::IRRADIANCE_MAP);

    prefiltered_cubemap->use(Renderer::Impl::TextureBindings::PREFILTER_MAP);
    shader->setInt("prefilter_map", Renderer::Impl::TextureBindings::PREFILTER_MAP);

    lut->use(Renderer::Impl::TextureBindings::LUT_TEXTURE);
    shader->setInt("lut", Renderer::Impl::TextureBindings::LUT_TEXTURE);

    shader->setInt("use_ibl", has_skybox);
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
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    ShaderManager::addShaderFromName("base");
    ShaderManager::addShaderFromName("flat");
    ShaderManager::addShaderFromName("instanced");
    ShaderManager::addShaderFromName("edge");
    ShaderManager::addShaderFromName("circle");
    ShaderManager::addShaderFromName("screen");
    ShaderManager::addShaderFromName("cylinder_edge");
    ShaderManager::addShaderFromName("equirectangularToCubemap");
    ShaderManager::addShaderFromName("skybox");
    ShaderManager::addShaderFromName("cubeMapConvolution");
    ShaderManager::addShaderFromName("prefilter_cubemap");
    ShaderManager::addShaderFromName("vrScreen");
    ShaderManager::addComputerShaderFromName("white_noise_2D");
    ShaderManager::addComputerShaderFromName("white_noise_3D");
    ShaderManager::addComputerShaderFromName("worly_noise_2D");
    ShaderManager::addComputerShaderFromName("worly_noise_3D");
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

    ++s_renderer->impl->frame_counter;
}

void Renderer::setClearColor(const glm::vec4& color)
{
    s_renderer->impl->clear_color = color;
    glClearColor(color.r, color.g, color.b, color.a);
}

glm::vec4 Renderer::getClearColor()
{
    return s_renderer->impl->clear_color;
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

void Renderer::setDefaultViewport()
{
    setViewport(0, 0, getFramebuffer()->width(), getFramebuffer()->height());
}

void Renderer::setSkybox(const atcg::ref_ptr<Image>& skybox)
{
    bool culling                 = s_renderer->impl->culling_enabled;
    s_renderer->impl->has_skybox = true;
    toggleCulling(false);
    atcg::ref_ptr<PerspectiveCamera> capture_cam = atcg::make_ref<atcg::PerspectiveCamera>(1.0f);
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

    s_renderer->impl->skybox_texture = atcg::Texture2D::create(skybox);

    uint32_t current_fbo = atcg::Framebuffer::currentFramebuffer();
    int old_viewport[4];
    glGetIntegerv(GL_VIEWPORT, old_viewport);

    // * Create a cubemap from the equirectangular map
    {
        atcg::ref_ptr<Shader> equirect_shader = ShaderManager::getShader("equirectangularToCubemap");
        float width                           = s_renderer->impl->skybox_cubemap->width();
        float height                          = s_renderer->impl->skybox_cubemap->height();
        Framebuffer captureFBO(width, height);
        captureFBO.attachDepth();

        glViewport(0, 0, width, height);    // don't forget to configure the viewport to the capture dimensions.
        captureFBO.use();

        equirect_shader->use();
        s_renderer->impl->skybox_texture->use(Renderer::Impl::TextureBindings::COUNT);
        equirect_shader->setInt("equirectangularMap", Renderer::Impl::TextureBindings::COUNT);
        for(unsigned int i = 0; i < 6; ++i)
        {
            capture_cam->setView(captureViews[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER,
                                   GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                   s_renderer->impl->skybox_cubemap->getID(),
                                   0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            draw(s_renderer->impl->cube, capture_cam, glm::mat4(1), glm::vec3(1), equirect_shader);
            // renderCube();    // renders a 1x1 cube
        }

        s_renderer->impl->skybox_cubemap->generateMipmaps();
    }

    // * Convolution of cube map for irradiance map
    {
        atcg::ref_ptr<Shader> cubeconv_shader = ShaderManager::getShader("cubeMapConvolution");
        float width                           = s_renderer->impl->irradiance_cubemap->width();
        float height                          = s_renderer->impl->irradiance_cubemap->height();
        Framebuffer captureFBO(width, height);
        captureFBO.attachDepth();

        glViewport(0, 0, width, height);    // don't forget to configure the viewport to the capture dimensions.
        captureFBO.use();

        cubeconv_shader->use();
        s_renderer->impl->skybox_cubemap->use(Renderer::Impl::TextureBindings::COUNT);
        cubeconv_shader->setInt("skybox", Renderer::Impl::TextureBindings::COUNT);
        for(unsigned int i = 0; i < 6; ++i)
        {
            capture_cam->setView(captureViews[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER,
                                   GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                   s_renderer->impl->irradiance_cubemap->getID(),
                                   0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            draw(s_renderer->impl->cube, capture_cam, glm::mat4(1), glm::vec3(1), cubeconv_shader);
            // renderCube();    // renders a 1x1 cube
        }
    }

    // * Prefilter environment map
    {
        atcg::ref_ptr<Shader> prefilter_shader = ShaderManager::getShader("prefilter_cubemap");
        float width                            = s_renderer->impl->prefiltered_cubemap->width();
        float height                           = s_renderer->impl->prefiltered_cubemap->height();

        prefilter_shader->use();
        prefilter_shader->setInt("skybox", Renderer::Impl::TextureBindings::COUNT);
        unsigned int max_mip_levels = 5;
        for(unsigned int mip = 0; mip < max_mip_levels; ++mip)
        {
            unsigned int mip_width  = s_renderer->impl->prefiltered_cubemap->width() * std::pow(0.5, mip);
            unsigned int mip_height = s_renderer->impl->prefiltered_cubemap->height() * std::pow(0.5, mip);

            // Recreate captureFBO with new resolution
            Framebuffer captureFBO(mip_width, mip_height);
            captureFBO.attachDepth();
            captureFBO.use();

            s_renderer->impl->skybox_cubemap->use(Renderer::Impl::TextureBindings::COUNT);

            glViewport(0, 0, mip_width, mip_height);

            float roughness = (float)mip / (float)(max_mip_levels - 1);
            prefilter_shader->setFloat("roughness", roughness);
            for(unsigned int i = 0; i < 6; ++i)
            {
                capture_cam->setView(captureViews[i]);
                glFramebufferTexture2D(GL_FRAMEBUFFER,
                                       GL_COLOR_ATTACHMENT0,
                                       GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                       s_renderer->impl->prefiltered_cubemap->getID(),
                                       mip);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                draw(s_renderer->impl->cube, capture_cam, glm::mat4(1), glm::vec3(1), prefilter_shader);
                // renderCube();    // renders a 1x1 cube
            }
        }
    }

    Framebuffer::bindByID(current_fbo);
    setViewport(old_viewport[0], old_viewport[1], old_viewport[2], old_viewport[3]);
    toggleCulling(culling);
}

bool Renderer::hasSkybox()
{
    return s_renderer->impl->has_skybox;
}

void Renderer::removeSkybox()
{
    s_renderer->impl->has_skybox = false;
}

atcg::ref_ptr<Texture2D> Renderer::getSkyboxTexture()
{
    return s_renderer->impl->skybox_texture;
}

void Renderer::resize(const uint32_t& width, const uint32_t& height)
{
    setViewport(0, 0, width, height);
    s_renderer->impl->screen_fbo = atcg::make_ref<Framebuffer>(width, height);
    s_renderer->impl->screen_fbo->attachColor();
    TextureSpecification spec_int;
    spec_int.width  = width;
    spec_int.height = height;
    spec_int.format = TextureFormat::RINT;
    s_renderer->impl->screen_fbo->attachTexture(Texture2D::create(spec_int));
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
    s_renderer->impl->culling_enabled = enable;
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

uint32_t Renderer::getFrameCounter()
{
    return s_renderer->impl->frame_counter;
}

void Renderer::draw(const atcg::ref_ptr<Graph>& mesh,
                    const atcg::ref_ptr<Camera>& camera,
                    const glm::mat4& model,
                    const glm::vec3& color,
                    const atcg::ref_ptr<Shader>& shader,
                    DrawMode draw_mode)
{
    mesh->unmapAllPointers();
    switch(draw_mode)
    {
        case ATCG_DRAW_MODE_TRIANGLE:
        {
            shader->setInt("entityID", -1);
            s_renderer->impl->setMaterial(s_renderer->impl->standard_material, shader);
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
            shader->setInt("entityID", -1);
            s_renderer->impl->setMaterial(s_renderer->impl->standard_material, shader);
            s_renderer->impl
                ->drawVAO(mesh->getVerticesArray(), camera, color, shader, model, GL_POINTS, mesh->n_vertices());
        }
        break;
        case ATCG_DRAW_MODE_POINTS_SPHERE:
        {
            shader->setInt("entityID", -1);
            s_renderer->impl->setMaterial(s_renderer->impl->standard_material, shader);
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
            auto edge_shader = ShaderManager::getShader("edge");
            edge_shader->setInt("entityID", -1);
            s_renderer->impl->setMaterial(s_renderer->impl->standard_material, edge_shader);
            atcg::ref_ptr<VertexBuffer> points = mesh->getVerticesBuffer();
            points->bindStorage(0);
            s_renderer->impl
                ->drawVAO(mesh->getEdgesArray(), camera, color, edge_shader, model, GL_POINTS, mesh->n_edges(), 1);
        }
        break;
        case ATCG_DRAW_MODE_EDGES_CYLINDER:
        {
            auto edge_shader = ShaderManager::getShader("cylinder_edge");
            edge_shader->setInt("entityID", -1);
            s_renderer->impl->setMaterial(s_renderer->impl->standard_material, edge_shader);
            s_renderer->impl
                ->drawGrid(mesh->getVerticesBuffer(), mesh->getEdgesBuffer(), edge_shader, camera, model, color);
        }
        break;
        case ATCG_DRAW_MODE_INSTANCED:
        {
            shader->setInt("entityID", -1);
            s_renderer->impl->setMaterial(s_renderer->impl->standard_material, shader);
            atcg::ref_ptr<VertexArray> vao_mesh      = mesh->getVerticesArray();
            atcg::ref_ptr<VertexBuffer> instance_vbo = vao_mesh->peekVertexBuffer();
            uint32_t n_instances                     = instance_vbo->size() / instance_vbo->getLayout().getStride();
            s_renderer->impl
                ->drawVAO(vao_mesh, camera, color, shader, model, GL_TRIANGLES, mesh->n_vertices(), n_instances);
        }
        break;
    }
}

// drawEntity
void Renderer::draw(Entity entity, const atcg::ref_ptr<Camera>& camera)
{
    if(entity.hasComponent<CustomRenderComponent>())
    {
        CustomRenderComponent renderer = entity.getComponent<CustomRenderComponent>();
        renderer.callback(entity, camera);
    }

    if(entity.hasAnyComponent<MeshRenderComponent,
                              PointRenderComponent,
                              PointSphereRenderComponent,
                              EdgeRenderComponent,
                              EdgeCylinderRenderComponent,
                              InstanceRenderComponent>())
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
    }
    else
    {
        return;
    }

    uint32_t entity_id           = (uint32_t)entity._entity_handle;
    TransformComponent transform = entity.getComponent<TransformComponent>();
    GeometryComponent geometry   = entity.getComponent<GeometryComponent>();

    if(!geometry.graph)
    {
        ATCG_WARN("Entity does have geometry component but mesh is empty");
        return;
    }

    geometry.graph->unmapAllPointers();
    if(entity.hasComponent<MeshRenderComponent>())
    {
        MeshRenderComponent renderer = entity.getComponent<MeshRenderComponent>();


        if(renderer.visible)
        {
            s_renderer->impl->setMaterial(renderer.material, renderer.shader);
            renderer.shader->setInt("entityID", entity_id);
            s_renderer->impl->drawVAO(geometry.graph->getVerticesArray(),
                                      camera,
                                      glm::vec3(1),
                                      renderer.shader,
                                      transform.getModel(),
                                      GL_TRIANGLES,
                                      geometry.graph->n_vertices());
        }
    }

    if(entity.hasComponent<PointRenderComponent>())
    {
        PointRenderComponent renderer = entity.getComponent<PointRenderComponent>();
        if(renderer.visible)
        {
            s_renderer->impl->setMaterial(s_renderer->impl->standard_material, renderer.shader);
            renderer.shader->setInt("entityID", entity_id);
            setPointSize(renderer.point_size);
            s_renderer->impl->drawVAO(geometry.graph->getVerticesArray(),
                                      camera,
                                      renderer.color,
                                      renderer.shader,
                                      transform.getModel(),
                                      GL_POINTS,
                                      geometry.graph->n_vertices());
        }
    }

    if(entity.hasComponent<PointSphereRenderComponent>())
    {
        PointSphereRenderComponent renderer = entity.getComponent<PointSphereRenderComponent>();

        if(renderer.visible)
        {
            s_renderer->impl->setMaterial(renderer.material, renderer.shader);
            renderer.shader->setInt("entityID", entity_id);
            setPointSize(renderer.point_size);
            s_renderer->impl->drawPointCloudSpheres(geometry.graph->getVerticesArray()->peekVertexBuffer(),
                                                    camera,
                                                    transform.getModel(),
                                                    glm::vec3(1),
                                                    renderer.shader,
                                                    geometry.graph->n_vertices());
        }
    }

    if(entity.hasComponent<EdgeRenderComponent>())
    {
        EdgeRenderComponent renderer = entity.getComponent<EdgeRenderComponent>();

        if(renderer.visible)
        {
            s_renderer->impl->setMaterial(s_renderer->impl->standard_material, ShaderManager::getShader("edge"));
            ShaderManager::getShader("edge")->setInt("entityID", entity_id);
            atcg::ref_ptr<VertexBuffer> points = geometry.graph->getVerticesBuffer();
            points->bindStorage(0);
            s_renderer->impl->drawVAO(geometry.graph->getEdgesArray(),
                                      camera,
                                      renderer.color,
                                      ShaderManager::getShader("edge"),
                                      transform.getModel(),
                                      GL_POINTS,
                                      geometry.graph->n_edges(),
                                      1);
        }
    }

    if(entity.hasComponent<EdgeCylinderRenderComponent>())
    {
        EdgeCylinderRenderComponent renderer = entity.getComponent<EdgeCylinderRenderComponent>();

        if(renderer.visible)
        {
            auto& shader = ShaderManager::getShader("cylinder_edge");
            s_renderer->impl->setMaterial(renderer.material, shader);
            shader->setInt("entityID", entity_id);
            shader->setFloat("edge_radius", renderer.radius);
            s_renderer->impl->drawGrid(geometry.graph->getVerticesBuffer(),
                                       geometry.graph->getEdgesBuffer(),
                                       ShaderManager::getShader("cylinder_edge"),
                                       camera,
                                       transform.getModel(),
                                       glm::vec3(1));
        }
    }

    if(entity.hasComponent<InstanceRenderComponent>())
    {
        InstanceRenderComponent renderer = entity.getComponent<InstanceRenderComponent>();

        if(renderer.visible)
        {
            if(geometry.graph->getVerticesArray()->peekVertexBuffer() != renderer.instance_vbo)
            {
                geometry.graph->getVerticesArray()->pushInstanceBuffer(renderer.instance_vbo);
            }

            auto instance_shader = ShaderManager::getShader("instanced");
            instance_shader->setInt("entityID", entity_id);
            s_renderer->impl->setMaterial(renderer.material, instance_shader);
            atcg::ref_ptr<VertexArray> vao_mesh      = geometry.graph->getVerticesArray();
            atcg::ref_ptr<VertexBuffer> instance_vbo = vao_mesh->peekVertexBuffer();
            uint32_t n_instances                     = instance_vbo->size() / instance_vbo->getLayout().getStride();
            s_renderer->impl->drawVAO(vao_mesh,
                                      camera,
                                      glm::vec3(1),
                                      instance_shader,
                                      transform.getModel(),
                                      GL_TRIANGLES,
                                      geometry.graph->n_vertices(),
                                      n_instances);
        }
    }
}

// drawScene
void Renderer::draw(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera)
{
    // TODO: Just raw opengl rendering code here
    if(s_renderer->impl->has_skybox)
    {
        glDepthMask(GL_FALSE);
        glDepthFunc(GL_LEQUAL);
        bool culling = s_renderer->impl->culling_enabled;
        toggleCulling(false);
        ShaderManager::getShader("skybox")->use();
        ShaderManager::getShader("skybox")->setInt("skybox", Renderer::Impl::TextureBindings::SKYBOX_TEXTURE);
        s_renderer->impl->skybox_cubemap->use(Renderer::Impl::TextureBindings::SKYBOX_TEXTURE);

        draw(s_renderer->impl->cube, camera, glm::mat4(1), glm::vec3(1), ShaderManager::getShader("skybox"));

        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);
        toggleCulling(culling);
    }

    const auto& view = scene->getAllEntitiesWith<atcg::TransformComponent>();

    for(auto e: view)
    {
        Entity entity(e, scene.get());
        Renderer::draw(entity, camera);
    }
}

void Renderer::drawCameras(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera)
{
    const auto& view = scene->getAllEntitiesWith<atcg::CameraComponent>();

    for(auto e: view)
    {
        Entity entity(e, scene.get());
        setLineSize(2.0f);
        uint32_t entity_id = (uint32_t)entity._entity_handle;
        atcg::ShaderManager::getShader("edge")->setInt("entityID", entity_id);
        atcg::CameraComponent& comp          = entity.getComponent<CameraComponent>();
        atcg::ref_ptr<PerspectiveCamera> cam = std::dynamic_pointer_cast<PerspectiveCamera>(comp.camera);
        float aspect_ratio                   = cam->getAspectRatio();
        glm::mat4 scale =
            glm::scale(glm::vec3(aspect_ratio, 1.0f, -0.5f / glm::tan(glm::radians(cam->getFOV()) / 2.0f)));
        glm::mat4 model = glm::inverse(cam->getView()) * scale;
        Renderer::draw(s_renderer->impl->camera_frustrum,
                       camera,
                       model,
                       comp.color,
                       atcg::ShaderManager::getShader("edge"),
                       atcg::DrawMode::ATCG_DRAW_MODE_EDGES);
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
        if(s_renderer->impl->sphere_has_instance)
        {
            vao_sphere->popVertexBuffer();
        }
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

void Renderer::drawCircle(const glm::vec3& position,
                          const float& radius,
                          const float& thickness,
                          const glm::vec3& color,
                          const atcg::ref_ptr<Camera>& camera)
{
    s_renderer->impl->quad_vao->use();
    const auto& shader = ShaderManager::getShader("circle");
    shader->setVec3("flat_color", color);
    shader->setFloat("radius", radius);
    shader->setFloat("thickness", thickness);
    shader->setVec3("position", position);
    if(camera)
    {
        shader->setMVP(glm::mat4(1), camera->getView(), camera->getProjection());
    }

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

void Renderer::drawCADGrid(const atcg::ref_ptr<Camera>& camera, const float& transparency_)
{
    float distance     = glm::abs(camera->getPosition().y);
    float current_size = s_renderer->impl->line_size;

    setLineSize(1.0f);

    auto& shader = atcg::ShaderManager::getShader("edge");
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

            draw(s_renderer->impl->grid,
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
    draw(s_renderer->impl->cross, camera, glm::mat4(1), glm::vec3(1), shader, atcg::DrawMode::ATCG_DRAW_MODE_EDGES);
    setLineSize(current_size);

    shader->setFloat("fall_off_edge", 1000.0f);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);
}

int Renderer::getEntityIndex(const glm::vec2& mouse)
{
    useScreenBuffer();
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    int pixelData;
    glReadPixels((int)mouse.x, (int)mouse.y, 1, 1, GL_RED_INTEGER, GL_INT, &pixelData);
    return pixelData;
}

void Renderer::screenshot(const atcg::ref_ptr<Scene>& scene,
                          const atcg::ref_ptr<Camera>& camera,
                          const uint32_t width,
                          const std::string& path)
{
    auto data = screenshot(scene, camera, width);

    Image img(data);

    img.store(path);
}

torch::Tensor
Renderer::screenshot(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera, const uint32_t width)
{
    atcg::ref_ptr<PerspectiveCamera> cam         = std::dynamic_pointer_cast<PerspectiveCamera>(camera);
    float height                                 = (float)width / cam->getAspectRatio();
    atcg::ref_ptr<Framebuffer> screenshot_buffer = atcg::make_ref<Framebuffer>((int)width, (int)height);
    screenshot_buffer->attachColor();
    screenshot_buffer->attachDepth();
    screenshot_buffer->complete();

    screenshot_buffer->use();
    atcg::Renderer::clear();
    atcg::Renderer::setViewport(0, 0, width, height);
    atcg::Renderer::draw(scene, cam);
    atcg::Renderer::getFramebuffer()->use();
    atcg::Renderer::setDefaultViewport();

    std::vector<uint8_t> buffer = getFrame(screenshot_buffer);

    auto data = screenshot_buffer->getColorAttachement(0)->getData(atcg::CPU);

    return data;
}

std::vector<uint8_t> Renderer::getFrame()
{
    return getFrame(s_renderer->impl->screen_fbo);
}

std::vector<uint8_t> Renderer::getFrame(const atcg::ref_ptr<Framebuffer>& fbo)
{
    auto frame      = fbo->getColorAttachement();
    uint32_t width  = frame->width();
    uint32_t height = frame->height();
    std::vector<uint8_t> buffer(width * height * 4);

    fbo->use();

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