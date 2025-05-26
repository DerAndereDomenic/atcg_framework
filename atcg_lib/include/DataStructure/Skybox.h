#pragma once

#include <Renderer/Texture.h>

namespace atcg
{
class Skybox
{
public:
    Skybox();

    Skybox(const atcg::ref_ptr<atcg::Texture2D>& skybox_texture);

    void setSkyboxTexture(const atcg::ref_ptr<atcg::Texture2D>& skybox_texture);

    ATCG_INLINE atcg::ref_ptr<atcg::Texture2D> getSkyboxTexture() const { return _skybox_texture; }

    ATCG_INLINE atcg::ref_ptr<atcg::TextureCube> getSkyboxCubeMap() const { return _skybox_cubemap; }

    ATCG_INLINE atcg::ref_ptr<atcg::TextureCube> getIrradianceMap() const { return _irradiance_cubemap; }

    ATCG_INLINE atcg::ref_ptr<atcg::TextureCube> getPrefilteredMap() const { return _prefiltered_cubemap; }

private:
    void _initTextures();

    atcg::ref_ptr<atcg::Texture2D> _skybox_texture;
    atcg::ref_ptr<atcg::TextureCube> _skybox_cubemap;
    atcg::ref_ptr<atcg::TextureCube> _irradiance_cubemap;
    atcg::ref_ptr<atcg::TextureCube> _prefiltered_cubemap;
};
}    // namespace atcg