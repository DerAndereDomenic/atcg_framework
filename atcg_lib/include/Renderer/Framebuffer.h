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
     * @brief Attach a texture to the framebuffer
     *
     * @param texture The texture to attach
     */
    void attachTexture(const atcg::ref_ptr<Texture2D>& texture);

    /**
     * @brief Add a depth attachement
     */
    void attachDepth();

    /**
     * @brief Attach a custom depth component
     *
     * @param depth_map The depth map component
     */
    void attachDepth(const atcg::ref_ptr<Texture2D>& depth_map);

    /**
     * @brief Get a color attachement
     *
     * @param slot The number of which attachement to use
     * @return The specified texture
     */
    ATCG_INLINE atcg::ref_ptr<Texture2D> getColorAttachement(const uint32_t& slot = 0) const
    {
        return _color_attachements[slot];
    }

    /**
     * @brief Get the depth attachement
     *
     * @return The depth texture
     */
    ATCG_INLINE atcg::ref_ptr<Texture2D> getDepthAttachement() const { return _depth_attachement; }

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
    static uint32_t s_current_fbo;
    uint32_t _ID;
    uint32_t _width, _height;
    std::vector<atcg::ref_ptr<Texture2D>> _color_attachements;
    atcg::ref_ptr<Texture2D> _depth_attachement;
};
}    // namespace atcg