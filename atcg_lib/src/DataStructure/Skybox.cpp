#include <DataStructure/Skybox.h>
#include <Renderer/Renderer.h>

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
    _skybox_texture = skybox_texture;
    Renderer::processSkybox(_skybox_texture, _skybox_cubemap, _irradiance_cubemap, _prefiltered_cubemap);
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