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
     * @brief Vertify that the framebuffer is complete
     *
     * @return True if it is complete, false otherwise
     */
    bool verify() const;

    /**
     * @brief Add a color attachement
     */
    void attachColor();

    /**
     * @brief Add a depth attachement
     */
    void attachDepth();

    /**
     * @brief Get a color attachement
     *
     * @param slot The number of which attachement to use
     * @return The specified texture
     */
    inline std::shared_ptr<Texture2D> getColorAttachement(const uint32_t& slot = 0) const
    {
        return _color_attachements[slot];
    }

    /**
     * @brief Get the depth attachement
     *
     * @return The depth texture
     */
    inline std::shared_ptr<Texture2D> getDepthAttachement() const { return _depth_attachement; }

private:
    friend class Renderer;
    static void useDefault();
    uint32_t _ID;
    uint32_t _width, _height;
    std::vector<std::shared_ptr<Texture2D>> _color_attachements;
    std::shared_ptr<Texture2D> _depth_attachement;
};
}    // namespace atcg