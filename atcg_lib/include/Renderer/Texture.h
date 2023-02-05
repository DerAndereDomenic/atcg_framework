#pragma once

#include <cstdint>
#include <memory>

namespace atcg
{
/**
 * @brief A class to model a texture
 */
class Texture2D
{
public:
    /**
     * @brief Create a RGBA color texture
     *
     * @param width The width
     * @param height The height
     */
    static std::shared_ptr<Texture2D> createColorTexture(uint32_t width, uint32_t height);

    /**
     * @brief Create a depth texture
     *
     * @param width The width
     * @param height The height
     */
    static std::shared_ptr<Texture2D> createDepthTexture(uint32_t width, uint32_t height);

    /**
     *  @brief Destructor
     */
    ~Texture2D();

    /**
     * @brief Get the width of the texture
     *
     * @return The width
     */
    inline uint32_t width() const { return _width; }

    /**
     * @brief Get the height of the texture
     *
     * @return The height
     */
    inline uint32_t height() const { return _height; }

    /**
     * @brief Get the id of the texture
     *
     * @return The id
     */
    inline uint32_t getID() const { return _ID; }

    /**
     * @brief Use this texture
     *
     * @param slot The used texture slot
     */
    void use(const uint32_t& slot = 0) const;

private:
    uint32_t _width, _height;
    uint32_t _ID;
};
}    // namespace atcg