#pragma once

#include <Renderer/Texture.h>

#include <vector>

namespace atcg
{

class GraphicsAPI;

/**
 * @brief The type of framebuffer texture
 */
enum class FramebufferTextureFormat
{
    TEXTURE_2D,
    TEXTURE_3D,
    TEXTURE_CUBE,
    TEXTURE_ARRAY,
    TEXTURE_CUBE_ARRAY,
    TEXTURE_2D_MULTISAMPLE

};

/**
 * @brief A framebuffer texture specification.
 * This consists of
 * * The definition of the texture
 * * If the texture is a depth map
 * * The format of the texture
 */
struct FramebufferTextureSpecification
{
    FramebufferTextureSpecification() = default;
    FramebufferTextureSpecification(TextureSpecification spec) : spec(spec) {}
    FramebufferTextureSpecification(TextureSpecification spec, FramebufferTextureFormat format)
        : spec(spec),
          format(format)
    {
    }

    FramebufferTextureSpecification(TextureSpecification spec, bool is_depth) : spec(spec), is_depth(is_depth) {}

    FramebufferTextureSpecification(TextureSpecification spec, FramebufferTextureFormat format, bool is_depth)
        : spec(spec),
          format(format),
          is_depth(is_depth)
    {
    }

    TextureSpecification spec;
    FramebufferTextureFormat format = FramebufferTextureFormat::TEXTURE_2D;
    bool is_depth                   = false;
};

/**
 * @brief A framebuffer specification consisting of
 * * The resolution of the framebuffer
 * * number of samples if MSAA is enabled
 * * The specifications of the attachements
 */
struct FramebufferSpecification
{
    FramebufferSpecification() = default;
    FramebufferSpecification(uint32_t width,
                             uint32_t height,
                             uint32_t num_samples,
                             std::initializer_list<FramebufferTextureSpecification> attachements)
        : width(width),
          height(height),
          num_samples(num_samples),
          attachements(attachements)
    {
    }

    FramebufferSpecification(uint32_t width,
                             uint32_t height,
                             uint32_t depth,
                             uint32_t num_samples,
                             std::initializer_list<FramebufferTextureSpecification> attachements)
        : width(width),
          height(height),
          depth(depth),
          num_samples(num_samples),
          attachements(attachements)
    {
    }

    uint32_t width       = 0;
    uint32_t height      = 0;
    uint32_t depth       = 0;
    uint32_t num_samples = 1;

    std::vector<FramebufferTextureSpecification> attachements;
};

/**
 * @brief Class to model a framebuffer
 */
class Framebuffer : public std::enable_shared_from_this<Framebuffer>
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
     * @brief Create a framebuffer from specification
     *
     * @param spec The specification
     */
    static atcg::ref_ptr<Framebuffer> create(const FramebufferSpecification& spec);

    /**
     * @brief Destructor
     */
    ~Framebuffer();

    /**
     * @brief Complete the Framebuffer. Should be called after all attachements where made
     *
     * @return True if it is complete, false otherwise
     */
    bool complete();

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
     * @brief Attach a face of a cube map as color attachement
     *
     * @param cube_map The cube map
     * @param face_index The face index (0-5)
     * @param mip_level The mip level
     */
    void attachCubeFace(const atcg::ref_ptr<TextureCube>& cube_map, uint32_t face_index, uint32_t mip_level = 0);

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
     * @brief Detach the last color attachement
     */
    void detachColor();

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
     * @brief Get number of color attachements
     *
     * @return Number of color attachements
     */
    ATCG_INLINE uint32_t numColorAttachements() const { return _color_attachements.size(); }

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
    static atcg::ref_ptr<Framebuffer> currentFramebuffer();


private:
    uint32_t _ID;
    uint32_t _width, _height;
    std::vector<atcg::ref_ptr<Texture>> _color_attachements;
    atcg::ref_ptr<Texture> _depth_attachement;

    /**
     * @brief Use the framebuffer
     */
    void bind();

    /**
     * @brief Use the default framebuffer
     */
    static void bindDefault();

    friend class GraphicsAPI;
};
}    // namespace atcg