#pragma once

#include <Renderer/Texture.h>

namespace atcg
{
/**
 * @brief A class to model a skybox
 */
class Skybox
{
public:
    /**
     * @brief Constructor that allocates empty textures
     */
    Skybox();

    /**
     * @brief Constructor that and computes textures based on the given equirectangular skybox texture.
     *
     * @param skybox_texture The equirectangular skybox
     */
    Skybox(const atcg::ref_ptr<atcg::Texture2D>& skybox_texture);

    /**
     * @brief Computes textures based on the given equirectangular skybox texture.
     *
     * @param skybox_texture The equirectangular skybox
     */
    void setSkyboxTexture(const atcg::ref_ptr<atcg::Texture2D>& skybox_texture);

    /**
     * @brief Get the equirectangular texture of the skybox
     *
     * @return The skybox texture
     */
    ATCG_INLINE atcg::ref_ptr<atcg::Texture2D> getSkyboxTexture() const { return _skybox_texture; }

    /**
     * @brief Get the skybox as cubemap
     *
     * @return The cubemap
     */
    ATCG_INLINE atcg::ref_ptr<atcg::TextureCube> getSkyboxCubeMap() const { return _skybox_cubemap; }

    /**
     * @brief Get the precomputed irradiance map for IBL
     *
     * @return The irradiance map
     */
    ATCG_INLINE atcg::ref_ptr<atcg::TextureCube> getIrradianceMap() const { return _irradiance_cubemap; }

    /**
     * @brief Get the prefiltered map for IBL.
     *
     * @return The prefiltered map
     */
    ATCG_INLINE atcg::ref_ptr<atcg::TextureCube> getPrefilteredMap() const { return _prefiltered_cubemap; }

private:
    void _initTextures();

    atcg::ref_ptr<atcg::Texture2D> _skybox_texture;
    atcg::ref_ptr<atcg::TextureCube> _skybox_cubemap;
    atcg::ref_ptr<atcg::TextureCube> _irradiance_cubemap;
    atcg::ref_ptr<atcg::TextureCube> _prefiltered_cubemap;
};
}    // namespace atcg