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
    REPEAT,
    // Border
    BORDER
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
     * @param mip_level The mip level
     *
     * @return The data
     */
    virtual torch::Tensor getData(const torch::Device& device = torch::Device(atcg::GPU),
                                  const uint32_t mip_level    = 0) const = 0;

    /**
     * @brief Get the width of the texture
     *
     * @return The width
     */
    ATCG_INLINE uint32_t width() const { return _spec.width; }

    /**
     * @brief Get the height of the texture
     *
     * @return The height
     */
    ATCG_INLINE uint32_t height() const { return _spec.height; }

    /**
     * @brief Get the depth of the texture
     *
     * @return The depth
     */
    ATCG_INLINE uint32_t depth() const { return _spec.depth; }

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
    ATCG_INLINE uint32_t getID() const { return _ID; }

    /**
     * @brief Get the texture specification.
     *
     * @return The specification
     */
    ATCG_INLINE TextureSpecification getSpecification() const { return _spec; }

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
     * @brief Create a deep copy of the texture
     *
     * @return The copy
     */
    virtual atcg::ref_ptr<Texture> clone() const = 0;

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
     * @param array_idx If *this is a TextureArray, this index is used to access a specific layer
     *
     * @return The pointer
     */
    atcg::textureArray getTextureArray(const uint32_t mip_level = 0, const uint32_t array_idx = 0) const;

    /**
     * @brief Get the underlying data as a cudaTextureObject_t.
     * This only returns a valid cudaTextureObject_t if the CUDA backend is enabled. Otherwise this will return the
     * buffer mapped to host space.
     *
     * @note This function should be called every frame and the pointer should not be cached by the application. OpenGL
     * is allowed to move buffers in memory. Therefore, the pointer might no longer be valid. The underlying resource
     * gets mapped and unmapped automatically. Every call to use(), bindStorage() or setData() invalidates the pointer.
     * If the buffer does not get explicitly binded again (because a VertexArray for example only points to this
     * buffer), the client has to manually unmap the pointers using unmapPointers() before any further rendering calls
     * can be done.
     *
     * @param mip_level The mip level
     * @param border_color The border color
     * @param normalized_coords If the coordinates should be normalized between 0 and 1
     * @param normalized_float If the texture should be read as normalized floats
     * @param array_idx If *this is a TextureArray, this index is used to access a specific layer
     *
     * @return The pointer
     */
    atcg::textureObject getTextureObject(const uint32_t mip_level      = 0,
                                         const glm::vec4& border_color = glm::vec4(0),
                                         const bool normalized_coords  = true,
                                         const bool normalized_float   = true,
                                         const uint32_t array_idx      = 0) const;

    /**
     * @brief Get the underlying data as a cudaSurfaceObject_t.
     * This only returns a valid cudaSurfaceObject_t if the CUDA backend is enabled. Otherwise this will return the
     * buffer mapped to host space.
     *
     * @note This function should be called every frame and the pointer should not be cached by the application. OpenGL
     * is allowed to move buffers in memory. Therefore, the pointer might no longer be valid. The underlying resource
     * gets mapped and unmapped automatically. Every call to use(), bindStorage() or setData() invalidates the pointer.
     * If the buffer does not get explicitly binded again (because a VertexArray for example only points to this
     * buffer), the client has to manually unmap the pointers using unmapPointers() before any further rendering calls
     * can be done.
     *
     * @param mip_level The mip level
     * @param array_idx If *this is a TextureArray, this index is used to access a specific layer
     *
     * @return The pointer
     */
    atcg::surfaceObject getSurfaceObject(const uint32_t mip_level = 0, const uint32_t array_idx = 0) const;

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

    /**
     * @brief Fill a texture with a specified value.
     *
     * @param value The fill value
     */
    void fill(void* value);

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
     * @param data The image
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<Texture2D> create(const torch::Tensor& img);

    /**
     * @brief Create a 2D texture.
     *
     * @param data The image
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
     * @param mip_level The mip level
     *
     * @return The data
     */
    virtual torch::Tensor getData(const torch::Device& device = torch::Device(atcg::GPU),
                                  const uint32_t mip_level    = 0) const override;

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

    /**
     * @brief Create a deep copy of the texture
     *
     * @return The copy
     */
    virtual atcg::ref_ptr<Texture> clone() const override;
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
     * @param data The image
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<Texture3D> create(const torch::Tensor& img);

    /**
     * @brief Create a 3D texture.
     *
     * @param data The image
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
     * @param mip_level The mip level
     *
     * @return The data
     */
    virtual torch::Tensor getData(const torch::Device& device = torch::Device(atcg::GPU),
                                  const uint32_t mip_level    = 0) const override;

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

    /**
     * @brief Create a deep copy of the texture
     *
     * @return The copy
     */
    virtual atcg::ref_ptr<Texture> clone() const override;
};

/**
 * @brief A class to model a cube map
 */
class TextureCube : public Texture
{
public:
    /**
     * @brief Create an empty cube map texture.
     *
     * @param spec The texture specification
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<TextureCube> create(const TextureSpecification& spec);

    /**
     * @brief Create a cube map texture from a tensor.
     *
     * @param data The image of shape (6, height, width, channels)
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<TextureCube> create(const torch::Tensor& img);

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
     * This function will always copy data between CPU and GPU via opengl. After construction, the tensor will be copied
     * onto the specified device.
     *
     * @param device The device
     * @param mip_level The mip level
     *
     * @return The data
     */
    virtual torch::Tensor getData(const torch::Device& device = torch::Device(atcg::GPU),
                                  const uint32_t mip_level    = 0) const override;

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

    /**
     * @brief Create a deep copy of the texture
     *
     * @return The copy
     */
    virtual atcg::ref_ptr<Texture> clone() const override;
};

/**
 * @brief A class to model a texture
 */
class TextureArray : public Texture
{
public:
    /**
     * @brief Create an empty texture array.
     *
     * @param spec The texture specification
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<TextureArray> create(const TextureSpecification& spec);

    /**
     * @brief Create a texture array.
     *
     * @param data The texture data
     * @param spec The texture specification
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<TextureArray> create(const void* data, const TextureSpecification& spec);

    /**
     * @brief Create a texture array.
     *
     * @param data The image
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<TextureArray> create(const torch::Tensor& img);

    /**
     * @brief Create a texture array.
     *
     * @param data The image
     * @param spec The texture specification
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<TextureArray> create(const torch::Tensor& img, const TextureSpecification& spec);

    /**
     *  @brief Destructor
     */
    virtual ~TextureArray();

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
     * @param mip_level The mip level
     *
     * @return The data
     */
    virtual torch::Tensor getData(const torch::Device& device = torch::Device(atcg::GPU),
                                  const uint32_t mip_level    = 0) const override;

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

    /**
     * @brief Create a deep copy of the texture
     *
     * @return The copy
     */
    virtual atcg::ref_ptr<Texture> clone() const override;
};

/**
 * @brief A class to model an array cube map
 */
class TextureCubeArray : public Texture
{
public:
    /**
     * @brief Create an empty cube map texture.
     *
     * @param spec The texture specification
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<TextureCubeArray> create(const TextureSpecification& spec);

    /**
     * @brief Create a cube map texture from a tensor.
     *
     * @param data The image of shape (n_layer, 6, height, width, channels)
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<TextureCubeArray> create(const torch::Tensor& img);

    /**
     *  @brief Destructor
     */
    virtual ~TextureCubeArray();

    /**
     * @brief Set the data of the cube map texture.
     * The tensor is supposed to be a (n_layer, 6, spec.height, spec.width, channels) tensor for each cubemap side for
     * each layer. This function will always transfer the data to opengl via a host to device upload. If the tensor is
     * on the GPU, it will be copied to host first.
     *
     * @param data The data
     */
    virtual void setData(const torch::Tensor& data) override;

    /**
     * @brief Get the data in the texture.
     * This function will always copy data between CPU and GPU via opengl. After construction, the tensor will be copied
     * onto the specified device.
     *
     * @param device The device
     * @param mip_level The mip level
     *
     * @return The data
     */
    virtual torch::Tensor getData(const torch::Device& device = torch::Device(atcg::GPU),
                                  const uint32_t mip_level    = 0) const override;

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

    /**
     * @brief Create a deep copy of the texture
     *
     * @return The copy
     */
    virtual atcg::ref_ptr<Texture> clone() const override;
};

/**
 * @brief This class is used to model a multi sampled texture used for anti aliasing.
 * It can only be manipulated by rendering to it. Setting or accessing data directly is not possible.
 */
class Texture2DMultiSample : public Texture
{
public:
    /**
     * @brief Create an empty multisampled 2D texture.
     *
     * @param num_samples The number of samples
     * @param spec The texture specification
     *
     * @return The resulting texture
     */
    static atcg::ref_ptr<Texture2DMultiSample> create(uint32_t num_samples, const TextureSpecification& spec);

    /**
     *  @brief Destructor
     */
    virtual ~Texture2DMultiSample();

    /**
     * @brief Set the data of the texture.
     * @note This function is a NoOp as it is not possible to set data to a multisampled texture
     *
     * @param data The data
     */
    virtual void setData(const torch::Tensor& data) override;

    /**
     * @brief Get the data in the texture.
     * @note This function is a NoOp as it is not possible to access data of a multisampled texture
     *
     * @param device The device
     * @param mip_level The mip level
     *
     * @return The data
     */
    virtual torch::Tensor getData(const torch::Device& device = torch::Device(atcg::GPU),
                                  const uint32_t mip_level    = 0) const override;

    /**
     * @brief Use this texture
     *
     * @param slot The used texture slot
     */
    virtual void use(const uint32_t& slot = 0) const override;

    /**
     * @brief Generate mipmap levels
     * @note This function is a NoOp as it is not possible to generate mipmaps of a multisampled texture.
     */
    virtual void generateMipmaps() override;

    /**
     * @brief Create a deep copy of the texture
     * @note This function is a NoOp as it is not possible to directly copy multi sample texture data.
     * Use framebuffers instead
     *
     * @return nullptr
     */
    virtual atcg::ref_ptr<Texture> clone() const override;
};

}    // namespace atcg