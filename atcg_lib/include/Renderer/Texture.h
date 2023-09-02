#pragma once

#include <Core/Memory.h>

#include <cstdint>
namespace atcg
{

/**
 * @brief The format of the texture.
 */
enum class TextureFormat
{
    // RGBA unsigned byte color texture.
    RGBA,
    // Red channel unsigned int texture.
    RINT,
    // Red channel byte texture.
    RINT8,
    // Red channel float 32 texture.
    RFLOAT,
    // Depth texture.
    DEPTH
};

/**
 * @brief The texture wrap mode.
 */
enum class TextureWrapMode
{
    // Extend the texture by the border pixel values
    CLAMP_TO_EDGE,
    // Repeat the texture
    REPEAT
};

/**
 * @brief The texture filter mode.
 */
enum class TextureFilterMode
{
    // Nearest neighbor filter.
    NEAREST,
    // Linear interpolation.
    LINEAR
};

/**
 * @brief The texture sampler.
 */
struct TextureSampler
{
    TextureWrapMode wrap_mode     = TextureWrapMode::REPEAT;
    TextureFilterMode filter_mode = TextureFilterMode::LINEAR;
};

struct TextureSpecification
{
    TextureFormat format   = TextureFormat::RGBA;
    TextureSampler sampler = {};
    uint32_t width         = 0;
    uint32_t height        = 0;
    uint32_t depth         = 0;
};

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
     * @brief Set the data of the texture.
     *
     * @param data The data
     */
    virtual void setData(const void* data) = 0;

    /**
     * @brief Get the width of the texture
     *
     * @return The width
     */
    inline uint32_t width() const { return _spec.width; }

    /**
     * @brief Get the height of the texture
     *
     * @return The height
     */
    inline uint32_t height() const { return _spec.height; }

    /**
     * @brief Get the depth of the texture
     *
     * @return The depth
     */
    inline uint32_t depth() const { return _spec.depth; }

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
    virtual void use(const uint32_t& slot = 0) const = 0;

    /**
     * @brief Use this texture as output in a compute shader
     *
     * @param slot The used texture slot
     */
    void useForCompute(const uint32_t& slot = 0) const;

protected:
    uint32_t _ID;
    TextureSpecification _spec;
};

/**
 * @brief A class to model a texture
 */
class Texture2D : public Texture
{
public:
    /**
     * @brief Create an empty 2D texture.
     *
     * @param spec The texture specification
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<Texture2D> create(const TextureSpecification& spec);

    /**
     * @brief Create a 2D texture.
     *
     * @param data The image data
     * @param spec The texture specification
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<Texture2D> create(const void* data, const TextureSpecification& spec);

    /**
     *  @brief Destructor
     */
    virtual ~Texture2D();

    /**
     * @brief Set the data of the texture.
     *
     * @param data The data
     */
    virtual void setData(const void* data) override;

    /**
     * @brief Use this texture
     *
     * @param slot The used texture slot
     */
    virtual void use(const uint32_t& slot = 0) const override;
};

/**
 * @brief A class to model a texture
 */
class Texture3D : public Texture
{
public:
    /**
     * @brief Create an empty 3D texture.
     *
     * @param spec The texture specification
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<Texture3D> create(const TextureSpecification& spec);

    /**
     * @brief Create a 3D texture.
     *
     * @param data The texture data
     * @param spec The texture specification
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<Texture3D> create(const void* data, const TextureSpecification& spec);

    /**
     *  @brief Destructor
     */
    virtual ~Texture3D();

    /**
     * @brief Set the data of the texture.
     *
     * @param data The data
     */
    virtual void setData(const void* data) override;

    /**
     * @brief Use this texture
     *
     * @param slot The used texture slot
     */
    virtual void use(const uint32_t& slot = 0) const override;
};
}    // namespace atcg