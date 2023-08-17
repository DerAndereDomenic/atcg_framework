#pragma once

#include <Core/Memory.h>

#include <cstdint>
namespace atcg
{

/**
 * @brief A class to model a texture
 */
class Texture
{
public:
    /**
     *  @brief Destructor
     */
    virtual ~Texture() {}

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
     * @brief Get the depth of the texture
     *
     * @return The depth
     */
    inline uint32_t depth() const { return _depth; }

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

    /**
     * @brief Use this texture as output in a compute shader
     *
     * @param slot The used texture slot
     */
    void useForCompute(const uint32_t& slot = 0) const;

protected:
    uint32_t _width, _height, _depth;
    uint32_t _ID;
    uint32_t _target;
};

/**
 * @brief A class to model a texture
 */
class Texture2D : public Texture
{
public:
    /**
     * @brief Create a RGBA color texture
     *
     * @param width The width
     * @param height The height
     */
    static atcg::ref_ptr<Texture2D> createColorTexture(uint32_t width, uint32_t height);

    /**
     * @brief Create a RGBA color texture
     *
     * @param data The texture data
     * @param width The width
     * @param height The height
     */
    static atcg::ref_ptr<Texture2D> createColorTexture(const glm::u8vec4* data, uint32_t width, uint32_t height);

    /**
     * @brief Create a depth texture
     *
     * @param width The width
     * @param height The height
     */
    static atcg::ref_ptr<Texture2D> createDepthTexture(uint32_t width, uint32_t height);

    /**
     * @brief Create a one channel int texture
     *
     * @param width The width
     * @param height The height
     */
    static atcg::ref_ptr<Texture2D> createIntTexture(uint32_t width, uint32_t height);

    /**
     * @brief Create a one channel float texture
     *
     * @param width The width
     * @param height The height
     */
    static atcg::ref_ptr<Texture2D> createFloatTexture(uint32_t width, uint32_t height);

    /**
     *  @brief Destructor
     */
    virtual ~Texture2D();
};

/**
 * @brief A class to model a texture
 */
class Texture3D : public Texture
{
public:
    /**
     * @brief Create a RGBA color texture
     *
     * @param width The width
     * @param height The height
     * @param depth The depth
     */
    static atcg::ref_ptr<Texture3D> createColorTexture(uint32_t width, uint32_t height, uint32_t depth);

    /**
     * @brief Create a one channel float texture
     *
     * @param width The width
     * @param height The height
     */
    static atcg::ref_ptr<Texture3D> createFloatTexture(uint32_t width, uint32_t height, uint32_t depth);

    /**
     *  @brief Destructor
     */
    virtual ~Texture3D();
};
}    // namespace atcg