#include <DataStructure/Skybox.h>
#include <Renderer/Renderer.h>

#include <Asset/AssetManagerSystem.h>

namespace atcg
{
Skybox::Skybox()
{
    _initTextures();
}

Skybox::Skybox(const atcg::ref_ptr<atcg::Texture2D>& skybox_texture)
{
    _initTextures();
    setSkyboxTexture(_skybox_texture);
}

void Skybox::setSkyboxTexture(const atcg::ref_ptr<atcg::Texture2D>& skybox_texture)
{
    auto cube       = AssetManager::getCubeMesh();
    _skybox_texture = skybox_texture;
    // Renderer::processSkybox(_skybox_texture, _skybox_cubemap, _irradiance_cubemap, _prefiltered_cubemap);

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

    uint32_t cubemap_id = Renderer::popTextureID();

    // * Create a cubemap from the equirectangular map
    {
        atcg::ref_ptr<Shader> equirect_shader = Renderer::getShaderManager()->getShader("equirectangularToCubemap");
        GraphicsPipeline pipeline             = GraphicsPipeline()
                                        .setPrimitiveTopology(PrimitiveTopology::ATCG_TRIANGLES)
                                        .setShader(equirect_shader)
                                        .setRasterizerState(RasterizerState().enableCulling(false));

        float width                           = _skybox_cubemap->width();
        float height                          = _skybox_cubemap->height();
        atcg::ref_ptr<Framebuffer> captureFBO = atcg::make_ref<Framebuffer>(width, height);
        captureFBO->attachDepth();
        GraphicsCommand::beginRenderPass(captureFBO);

        equirect_shader->setInt("equirectangularMap", cubemap_id);
        GraphicsCommand::bindTexture(cubemap_id, skybox_texture);
        for(unsigned int i = 0; i < 6; ++i)
        {
            capture_cam->setView(captureViews[i]);
            captureFBO->attachCubeFace(_skybox_cubemap, i, 0);
            GraphicsCommand::clear();

            Renderer::drawVAO(cube->getVerticesArray(), capture_cam, glm::mat4(1), pipeline, cube->n_vertices());
            captureFBO->detachColor();
        }

        _skybox_cubemap->generateMipmaps();
        GraphicsCommand::endRenderPass();
    }

    // * Convolution of cube map for irradiance map
    {
        atcg::ref_ptr<Shader> cubeconv_shader = Renderer::getShaderManager()->getShader("cubeMapConvolution");
        GraphicsPipeline pipeline             = GraphicsPipeline()
                                        .setPrimitiveTopology(PrimitiveTopology::ATCG_TRIANGLES)
                                        .setShader(cubeconv_shader)
                                        .setRasterizerState(RasterizerState().enableCulling(false));

        float width                           = _irradiance_cubemap->width();
        float height                          = _irradiance_cubemap->height();
        atcg::ref_ptr<Framebuffer> captureFBO = atcg::make_ref<Framebuffer>(width, height);
        captureFBO->attachDepth();

        GraphicsCommand::beginRenderPass(captureFBO);

        cubeconv_shader->setInt("skybox", cubemap_id);
        GraphicsCommand::bindTexture(cubemap_id, _skybox_cubemap);
        for(unsigned int i = 0; i < 6; ++i)
        {
            capture_cam->setView(captureViews[i]);
            captureFBO->attachCubeFace(_irradiance_cubemap, i, 0);
            GraphicsCommand::clear();

            Renderer::drawVAO(cube->getVerticesArray(), capture_cam, glm::mat4(1), pipeline, cube->n_vertices());
            captureFBO->detachColor();
        }
        GraphicsCommand::endRenderPass();
    }

    // * Prefilter environment map
    {
        atcg::ref_ptr<Shader> prefilter_shader = Renderer::getShaderManager()->getShader("prefilter_cubemap");
        float width                            = _prefiltered_cubemap->width();
        float height                           = _prefiltered_cubemap->height();

        prefilter_shader->setInt("skybox", cubemap_id);
        unsigned int max_mip_levels = 5;
        for(unsigned int mip = 0; mip < max_mip_levels; ++mip)
        {
            unsigned int mip_width  = _prefiltered_cubemap->width() * std::pow(0.5, mip);
            unsigned int mip_height = _prefiltered_cubemap->height() * std::pow(0.5, mip);

            // Recreate captureFBO with new resolution
            atcg::ref_ptr<Framebuffer> captureFBO = atcg::make_ref<Framebuffer>(mip_width, mip_height);
            captureFBO->attachDepth();

            GraphicsPipeline pipeline = GraphicsPipeline()
                                            .setPrimitiveTopology(PrimitiveTopology::ATCG_TRIANGLES)
                                            .setShader(prefilter_shader)
                                            .setRasterizerState(RasterizerState().enableCulling(false));

            GraphicsCommand::beginRenderPass(captureFBO);
            GraphicsCommand::bindTexture(cubemap_id, _skybox_cubemap);

            float roughness = (float)mip / (float)(max_mip_levels - 1);
            prefilter_shader->setFloat("roughness", roughness);

            for(unsigned int i = 0; i < 6; ++i)
            {
                capture_cam->setView(captureViews[i]);
                captureFBO->attachCubeFace(_prefiltered_cubemap, i, mip);
                GraphicsCommand::clear();

                Renderer::drawVAO(cube->getVerticesArray(), capture_cam, glm::mat4(1), pipeline, cube->n_vertices());
                captureFBO->detachColor();
            }
            GraphicsCommand::endRenderPass();
        }
    }

    Renderer::pushTextureID(cubemap_id);
}

void Skybox::_initTextures()
{
    TextureSpecification spec_skybox;
    spec_skybox.width               = 1024;
    spec_skybox.height              = 1024;
    spec_skybox.format              = TextureFormat::RGBAFLOAT;
    spec_skybox.sampler.wrap_mode   = TextureWrapMode::CLAMP_TO_EDGE;
    spec_skybox.sampler.filter_mode = TextureFilterMode::MIPMAP_LINEAR;
    _skybox_cubemap                 = atcg::TextureCube::create(spec_skybox);

    TextureSpecification spec_irradiance_cubemap;
    spec_irradiance_cubemap.width             = 32;
    spec_irradiance_cubemap.height            = 32;
    spec_irradiance_cubemap.format            = TextureFormat::RGBAFLOAT;
    spec_irradiance_cubemap.sampler.wrap_mode = TextureWrapMode::CLAMP_TO_EDGE;
    _irradiance_cubemap                       = atcg::TextureCube::create(spec_irradiance_cubemap);

    TextureSpecification spec_prefiltered_cubemap;
    spec_prefiltered_cubemap.width               = 128;
    spec_prefiltered_cubemap.height              = 128;
    spec_prefiltered_cubemap.format              = TextureFormat::RGBAFLOAT;
    spec_prefiltered_cubemap.sampler.wrap_mode   = TextureWrapMode::CLAMP_TO_EDGE;
    spec_prefiltered_cubemap.sampler.filter_mode = TextureFilterMode::MIPMAP_LINEAR;
    spec_prefiltered_cubemap.sampler.mip_map     = true;
    _prefiltered_cubemap                         = atcg::TextureCube::create(spec_prefiltered_cubemap);
}
}    // namespace atcg