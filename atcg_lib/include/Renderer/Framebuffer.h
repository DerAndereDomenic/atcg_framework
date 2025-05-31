#pragma once

#include <Renderer/Texture.h>

#include <vector>

namespace atcg
{
/**
 * @brief Class to model a framebuffer
 */
class Framebuffer
{
public:
    Framebuffer() = default;

    /**
     * @brief Create a framebuffer
     *
     * @param width The width
     * @param height The height
     */
    Framebuffer(uint32_t width, uint32_t height);

    /**
     * @brief Destructor
     */
    ~Framebuffer();

    /**
     * @brief Use the framebuffer
     */
    void use() const;

    /**
     * @brief Complete the Framebuffer. Should be called after all attachements where made
     *
     * @return True if it is complete, false otherwise
     */
    bool complete() const;

    /**
     * @brief Add a color attachement
     */
    void attachColor();

    /**
     * @brief Add a multi sampled color attachement
     */
    void attachColorMultiSample(uint32_t num_samples);

    /**
     * @brief Attach a texture to the framebuffer
     *
     * @param texture The texture to attach
     */
    void attachTexture(const atcg::ref_ptr<Texture>& texture);

    /**
     * @brief Add a depth attachement
     */
    void attachDepth();

    /**
     * @brief Add a multi sampled depth attachement
     */
    void attachDepthMultiSample(uint32_t num_samples);

    /**
     * @brief Attach a custom depth component
     *
     * @param depth_map The depth map component
     */
    void attachDepth(const atcg::ref_ptr<Texture>& depth_map);

    /**
     * @brief Blit two framebuffer together.
     * Copies the content of source into *this.
     *
     * @param source The source framebuffer to copy from
     * @param color If color information should be copied
     * @param depth If depth information should be copied
     */
    void blit(const atcg::ref_ptr<Framebuffer>& source, bool color = true, bool depth = true);

    /**
     * @brief Get a color attachement
     *
     * @param slot The number of which attachement to use
     * @return The specified texture
     */
    ATCG_INLINE atcg::ref_ptr<Texture> getColorAttachement(const uint32_t& slot = 0) const
    {
        return _color_attachements[slot];
    }

    /**
     * @brief Get the depth attachement
     *
     * @return The depth texture
     */
    ATCG_INLINE atcg::ref_ptr<Texture> getDepthAttachement() const { return _depth_attachement; }

    /**
     * @brief Get the ID of the framebuffer
     *
     * @return The ID
     */
    ATCG_INLINE uint32_t getID() const { return _ID; }

    /**
     * @brief Get the width of the framebuffer
     *
     * @return The width
     */
    ATCG_INLINE uint32_t width() const { return _width; }

    /**
     * @brief Get the height of the framebuffer
     *
     * @return The height
     */
    ATCG_INLINE uint32_t height() const { return _height; }

    /**
     * @brief Get the number of color attachements
     */
    ATCG_INLINE uint32_t getNumberAttachements() const { return _color_attachements.size(); }

    /**
     * @brief Get the currently bound fbo
     *
     * @return ID of the fbo
     */
    static uint32_t currentFramebuffer();

    /**
     * @brief Bind a specific framebuffer by ID
     *
     * @param ID The id to bind
     */
    static void bindByID(uint32_t fbo_id);

    /**
     * @brief Use the default framebuffer
     */
    static void useDefault();

private:
    uint32_t _ID;
    uint32_t _width, _height;
    std::vector<atcg::ref_ptr<Texture>> _color_attachements;
    atcg::ref_ptr<Texture> _depth_attachement;
};
}    // namespace atcg