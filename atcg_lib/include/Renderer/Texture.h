#pragma once

#include <Core/Memory.h>
#include <DataStructure/Image.h>
#include <DataStructure/TorchUtils.h>

#include <cstdint>
namespace atcg
{

/**
 * @brief The format of the texture.
 */
enum class TextureFormat
{
    // RG unsigned byte color texture.
    RG,
    // RGB unsigned byte color texture.
    RGB,
    // RGBA unsigned byte color texture.
    RGBA,
    // RG float color texture.
    RGFLOAT,
    // RGB float color texture.
    RGBFLOAT,
    // RGBA float color texture.
    RGBAFLOAT,
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
    LINEAR,
    // Trilinear interpolation using mipmaps (Needs to have a TextureSampler with mip_map = true)
    MIPMAP_LINEAR
};

/**
 * @brief The texture sampler.
 */
struct TextureSampler
{
    TextureWrapMode wrap_mode     = TextureWrapMode::REPEAT;
    TextureFilterMode filter_mode = TextureFilterMode::LINEAR;
    bool mip_map                  = false;
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
     * @brief Default constructor
     */
    Texture();

    /**
     *  @brief Destructor
     */
    virtual ~Texture();

    /**
     * @brief Set the data of the texture.
     * The tensor can be a host or device tensor if CUDA is enabled.
     * For CPU tensors a host-device memcpy is performed.
     * For Device Tensors a device-device copy is performed.
     *
     * @note A device-device memcpy can only be performed if the image has 1 or 4 channels. For three channel textures,
     * a host-device memcpy is required.
     *
     * @param data The data
     */
    virtual void setData(const torch::Tensor& data) = 0;

    /**
     * @brief Get the data in the texture.
     *
     * @note A device-device memcpy can only be performed if the image has 1 or 4 channels. For three channel textures,
     * a host-device memcpy is required.
     *
     * @param device The device
     *
     * @return The data
     */
    virtual torch::Tensor getData(const torch::Device& device = torch::Device(atcg::GPU)) const = 0;

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
     * @brief Get the number of channels
     *
     * @return The number of channels
     */
    uint32_t channels() const;

    /**
     * @brief Get if the texture is HDR
     *
     * @return True if the texture uses an internal float format
     */
    bool isHDR() const;

    /**
     * @brief Get the id of the texture
     *
     * @return The id
     */
    inline uint32_t getID() const { return _ID; }

    /**
     * @brief Get the texture specification.
     *
     * @return The specification
     */
    inline TextureSpecification getSpecification() const { return _spec; }

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

    /**
     * @brief Generate mipmap levels
     */
    virtual void generateMipmaps() = 0;

    /**
     * @brief Get the underlying data as a cudaArray.
     * This only returns a valid cudaArray if the CUDA backend is enabled. Otherwise this will return the buffer
     * mapped to host space.
     *
     * @note This function should be called every frame and the pointer should not be cached by the application. OpenGL
     * is allowed to move buffers in memory. Therefore, the pointer might no longer be valid. The underlying resource
     * gets mapped and unmapped automatically. Every call to use(), bindStorage() or setData() invalidates the pointer.
     * If the buffer does not get explicitly binded again (because a VertexArray for example only points to this
     * buffer), the client has to manually unmap the pointers using unmapPointers() before any further rendering calls
     * can be done.
     *
     * @param mip_level The mip level
     *
     * @return The pointer
     */
    atcg::textureArray getTextureArray(const uint32_t mip_level = 0) const;


    /**
     * @brief Check if the device pointer is mapped (valid).
     * @return True if the pointer is valid
     */
    bool isDeviceMapped() const;

    /**
     * @brief Unmaps and invalidates all device pointers used by the application
     */
    void unmapDevicePointers() const;

    /**
     * @brief Unmaps and invalidates all mapped pointers used by the application.
     */
    void unmapPointers() const;

protected:
    uint32_t _ID;
    TextureSpecification _spec;

    class Impl;
    std::unique_ptr<Impl> impl;
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
     * @brief Create a 2D texture.
     *
     * @param data The image
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<Texture2D> create(const atcg::ref_ptr<Image> img);

    /**
     * @brief Create a 2D texture.
     *
     * @param data The image
     * @param spec The texture specification
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<Texture2D> create(const atcg::ref_ptr<Image> img, const TextureSpecification& spec);

    /**
     * @brief Create a 2D texture.
     *
     * @param data The image (host data)
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<Texture2D> create(const torch::Tensor& img);

    /**
     * @brief Create a 2D texture.
     *
     * @param data The image (host data)
     * @param spec The texture specification
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<Texture2D> create(const torch::Tensor& img, const TextureSpecification& spec);

    /**
     *  @brief Destructor
     */
    virtual ~Texture2D();

    /**
     * @brief Set the data of the texture.
     * The tensor can be a host or device tensor if CUDA is enabled.
     * For CPU tensors a host-device memcpy is performed.
     * For Device Tensors a device-device copy is performed.
     *
     * @note A device-device memcpy can only be performed if the image has 1 or 4 channels. For three channel textures,
     * a host-device memcpy is required.
     *
     * @param data The data
     */
    virtual void setData(const torch::Tensor& data) override;

    /**
     * @brief Get the data in the texture.
     *
     * @note A device-device memcpy can only be performed if the image has 1 or 4 channels. For three channel textures,
     * a host-device memcpy is required.
     *
     * @param device The device
     *
     * @return The data
     */
    virtual torch::Tensor getData(const torch::Device& device = torch::Device(atcg::GPU)) const override;

    /**
     * @brief Use this texture
     *
     * @param slot The used texture slot
     */
    virtual void use(const uint32_t& slot = 0) const override;

    /**
     * @brief Generate mipmap levels
     */
    virtual void generateMipmaps() override;
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
     * @brief Create a 3D texture.
     *
     * @param data The image (host data)
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<Texture3D> create(const torch::Tensor& img);

    /**
     * @brief Create a 3D texture.
     *
     * @param data The image (host data)
     * @param spec The texture specification
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<Texture3D> create(const torch::Tensor& img, const TextureSpecification& spec);

    /**
     *  @brief Destructor
     */
    virtual ~Texture3D();

    /**
     * @brief Set the data of the texture.
     * The tensor can be a host or device tensor if CUDA is enabled.
     * For CPU tensors a host-device memcpy is performed.
     * For Device Tensors a device-device copy is performed.
     *
     * @note A device-device memcpy can only be performed if the image has 1 or 4 channels. For three channel textures,
     * a host-device memcpy is required.
     *
     * @param data The data
     */
    virtual void setData(const torch::Tensor& data) override;

    /**
     * @brief Get the data in the texture.
     *
     * @note A device-device memcpy can only be performed if the image has 1 or 4 channels. For three channel textures,
     * a host-device memcpy is required.
     *
     * @param device The device
     *
     * @return The data
     */
    virtual torch::Tensor getData(const torch::Device& device = torch::Device(atcg::GPU)) const override;

    /**
     * @brief Use this texture
     *
     * @param slot The used texture slot
     */
    virtual void use(const uint32_t& slot = 0) const override;

    /**
     * @brief Generate mipmap levels
     */
    virtual void generateMipmaps() override;
};

/**
 * @brief A class to model a cube map
 * @note This class has no setData or getData metods. It's also not possible to create cubemaps from existing data. It
 * is for now only filled by converting equicrectangular environment maps.
 */
class TextureCube : public Texture
{
public:
    /**
     * @brief Create an empty 2D texture.
     *
     * @param spec The texture specification
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<TextureCube> create(const TextureSpecification& spec);

    /**
     *  @brief Destructor
     */
    virtual ~TextureCube();

    /**
     * @brief Set the data of the cube map texture.
     * The tensor is supposed to be a (6, spec.height, spec.width, channels) tensor for each cubemap side.
     * This function will always transfer the data to opengl via a host to device upload. If the tensor is on the GPU,
     * it will be copied to host first.
     *
     * @param data The data
     */
    virtual void setData(const torch::Tensor& data) override;

    /**
     * @brief Get the data in the texture.
     *
     * @note A device-device memcpy can only be performed if the image has 1 or 4 channels. For three channel textures,
     * a host-device memcpy is required.
     *
     * @param device The device
     *
     * @return The data
     */
    virtual torch::Tensor getData(const torch::Device& device = torch::Device(atcg::GPU)) const override { return {}; }

    /**
     * @brief Use this texture
     *
     * @param slot The used texture slot
     */
    virtual void use(const uint32_t& slot = 0) const override;

    /**
     * @brief Generate mipmap levels
     */
    virtual void generateMipmaps() override;
};

}    // namespace atcg