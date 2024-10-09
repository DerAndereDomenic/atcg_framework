#include <Renderer/Texture.h>

#include <glad/glad.h>
#include <DataStructure/TorchUtils.h>

#ifdef ATCG_CUDA_BACKEND
    #include <cuda_gl_interop.h>
#endif

namespace atcg
{

namespace detail
{
GLint to2GLinternalFormat(TextureFormat format)
{
    switch(format)
    {
        case TextureFormat::RG:
        {
            return GL_RG;
        }
        case TextureFormat::RGB:
        {
            return GL_RGB;
        }
        case TextureFormat::RGBA:
        {
            return GL_RGBA;
        }
        case TextureFormat::RGFLOAT:
        {
            return GL_RG32F;
        }
        case TextureFormat::RGBFLOAT:
        {
            return GL_RGB32F;
        }
        case TextureFormat::RGBAFLOAT:
        {
            return GL_RGBA32F;
        }
        case TextureFormat::RINT:
        {
            return GL_R32I;
        }
        case TextureFormat::RINT8:
        {
            return GL_RED;
        }
        case TextureFormat::RFLOAT:
        {
            return GL_R32F;
        }
        case TextureFormat::DEPTH:
        {
            return GL_DEPTH_COMPONENT;
        }
        default:
        {
            ATCG_ERROR("Unknown TextureFormat {0}", (int)format);
            return -1;
        }
    }
}

GLenum toGLformat(TextureFormat format)
{
    switch(format)
    {
        case TextureFormat::RG:
        {
            return GL_RG;
        }
        case TextureFormat::RGB:
        {
            return GL_RGB;
        }
        case TextureFormat::RGBA:
        {
            return GL_RGBA;
        }
        case TextureFormat::RGFLOAT:
        {
            return GL_RG;
        }
        case TextureFormat::RGBFLOAT:
        {
            return GL_RGB;
        }
        case TextureFormat::RGBAFLOAT:
        {
            return GL_RGBA;
        }
        case TextureFormat::RINT:
        {
            return GL_RED_INTEGER;
        }
        case TextureFormat::RINT8:
        {
            return GL_RED;
        }
        case TextureFormat::RFLOAT:
        {
            return GL_RED;
        }
        case TextureFormat::DEPTH:
        {
            return GL_DEPTH_COMPONENT;
        }
        default:
        {
            ATCG_ERROR("Unknown TextureFormat {0}", (int)format);
            return -1;
        }
    }
}

GLenum toGLtype(TextureFormat format)
{
    switch(format)
    {
        case TextureFormat::RG:
        {
            return GL_UNSIGNED_BYTE;
        }
        case TextureFormat::RGB:
        {
            return GL_UNSIGNED_BYTE;
        }
        case TextureFormat::RGBA:
        {
            return GL_UNSIGNED_BYTE;
        }
        case TextureFormat::RGFLOAT:
        {
            return GL_FLOAT;
        }
        case TextureFormat::RGBFLOAT:
        {
            return GL_FLOAT;
        }
        case TextureFormat::RGBAFLOAT:
        {
            return GL_FLOAT;
        }
        case TextureFormat::RINT:
        {
            return GL_INT;
        }
        case TextureFormat::RINT8:
        {
            return GL_UNSIGNED_BYTE;
        }
        case TextureFormat::RFLOAT:
        {
            return GL_FLOAT;
        }
        case TextureFormat::DEPTH:
        {
            return GL_FLOAT;
        }
        default:
        {
            ATCG_ERROR("Unknown TextureFormat {0}", (int)format);
            return -1;
        }
    }
}

uint32_t num_channels(TextureFormat format)
{
    switch(format)
    {
        case TextureFormat::RG:
        {
            return 2;
        }
        case TextureFormat::RGB:
        {
            return 3;
        }
        case TextureFormat::RGBA:
        {
            return 4;
        }
        case TextureFormat::RGFLOAT:
        {
            return 2;
        }
        case TextureFormat::RGBFLOAT:
        {
            return 3;
        }
        case TextureFormat::RGBAFLOAT:
        {
            return 4;
        }
        case TextureFormat::RINT:
        {
            return 1;
        }
        case TextureFormat::RINT8:
        {
            return 1;
        }
        case TextureFormat::RFLOAT:
        {
            return 1;
        }
        case TextureFormat::DEPTH:
        {
            return 1;
        }
        default:
        {
            ATCG_ERROR("Unknown TextureFormat {0}", (int)format);
            return -1;
        }
    }
}

std::size_t toSize(TextureFormat format)
{
    switch(format)
    {
        case TextureFormat::RG:
        {
            return 2 * sizeof(uint8_t);
        }
        case TextureFormat::RGB:
        {
            return 3 * sizeof(uint8_t);
        }
        case TextureFormat::RGBA:
        {
            return 4 * sizeof(uint8_t);
        }
        case TextureFormat::RGFLOAT:
        {
            return 2 * sizeof(float);
        }
        case TextureFormat::RGBFLOAT:
        {
            return 3 * sizeof(float);
        }
        case TextureFormat::RGBAFLOAT:
        {
            return 4 * sizeof(float);
        }
        case TextureFormat::RINT:
        {
            return sizeof(uint32_t);
        }
        case TextureFormat::RINT8:
        {
            return sizeof(uint8_t);
        }
        case TextureFormat::RFLOAT:
        {
            return sizeof(float);
        }
        case TextureFormat::DEPTH:
        {
            return sizeof(float);
        }
        default:
        {
            ATCG_ERROR("Unknown TextureFormat {0}", (int)format);
            return -1;
        }
    }
}

std::size_t toChannelSize(TextureFormat format)
{
    switch(format)
    {
        case TextureFormat::RG:
        {
            return sizeof(uint8_t);
        }
        case TextureFormat::RGB:
        {
            return sizeof(uint8_t);
        }
        case TextureFormat::RGBA:
        {
            return sizeof(uint8_t);
        }
        case TextureFormat::RGFLOAT:
        {
            return sizeof(float);
        }
        case TextureFormat::RGBFLOAT:
        {
            return sizeof(float);
        }
        case TextureFormat::RGBAFLOAT:
        {
            return sizeof(float);
        }
        case TextureFormat::RINT:
        {
            return sizeof(uint32_t);
        }
        case TextureFormat::RINT8:
        {
            return sizeof(uint8_t);
        }
        case TextureFormat::RFLOAT:
        {
            return sizeof(float);
        }
        case TextureFormat::DEPTH:
        {
            return sizeof(float);
        }
        default:
        {
            ATCG_ERROR("Unknown TextureFormat {0}", (int)format);
            return -1;
        }
    }
}

GLint toGLWrapMode(TextureWrapMode wrap_mode)
{
    switch(wrap_mode)
    {
        case TextureWrapMode::CLAMP_TO_EDGE:
        {
            return GL_CLAMP_TO_EDGE;
        }
        case TextureWrapMode::REPEAT:
        {
            return GL_REPEAT;
        }
        case TextureWrapMode::BORDER:
        {
            return GL_CLAMP_TO_BORDER;
        }
        default:
        {
            ATCG_ERROR("Unknown TextureWrapMode {0}", (int)wrap_mode);
            return -1;
        }
    }
}

GLint toGLFilterMode(TextureFilterMode filter_mode)
{
    switch(filter_mode)
    {
        case TextureFilterMode::MIPMAP_LINEAR:
        {
            return GL_LINEAR_MIPMAP_LINEAR;
        }
        case TextureFilterMode::LINEAR:
        {
            return GL_LINEAR;
        }
        case TextureFilterMode::NEAREST:
        {
            return GL_NEAREST;
        }
        default:
        {
            ATCG_ERROR("Unknown TextureFilterMode {0}", (int)filter_mode);
            return -1;
        }
    }
}

#ifdef ATCG_CUDA_BACKEND
cudaTextureFilterMode toCUDATextureMode(TextureFilterMode filter_mode)
{
    switch(filter_mode)
    {
        case TextureFilterMode::MIPMAP_LINEAR:
        {
            return cudaFilterModeLinear;
        }
        case TextureFilterMode::LINEAR:
        {
            return cudaFilterModeLinear;
        }
        case TextureFilterMode::NEAREST:
        {
            return cudaFilterModePoint;
        }
        default:
        {
            ATCG_ERROR("Unknown TextureFilterMode {0}", (int)filter_mode);
            return cudaFilterModePoint;
        }
    }
}

cudaTextureAddressMode toCUDAAddressMode(TextureWrapMode wrap_mode)
{
    switch(wrap_mode)
    {
        case TextureWrapMode::CLAMP_TO_EDGE:
        {
            return cudaAddressModeClamp;
        }
        case TextureWrapMode::REPEAT:
        {
            return cudaAddressModeWrap;
        }
        case TextureWrapMode::BORDER:
        {
            return cudaAddressModeBorder;
        }
        default:
        {
            ATCG_ERROR("Unknown TextureWrapMode {0}", (int)wrap_mode);
            return cudaAddressModeClamp;
        }
    }
}
#endif

}    // namespace detail

class Texture::Impl
{
public:
    Impl();

    ~Impl();

    void initResource(GLuint ID, GLenum target);
    void deinitResource();

    void mapResourceDevice();
    void unmapResourceDevice();

#ifdef ATCG_CUDA_BACKEND
    cudaGraphicsResource* resource = nullptr;
#endif

    atcg::textureArray dev_ptr;
    atcg::textureObject texture_object;
    atcg::surfaceObject surface_object;

    GLuint ID           = 0;
    bool resource_ready = false;
    bool mapped_device  = false;
    bool texture_mapped = false;
    bool surface_mapped = false;
};

Texture::Impl::Impl() {}

Texture::Impl::~Impl() {}

Texture::Texture()
{
    impl = std::make_unique<Impl>();
}

Texture::~Texture()
{
    if(impl->resource_ready) impl->deinitResource();
}

void Texture::Impl::initResource(GLuint ID, GLenum target)
{
#ifdef ATCG_CUDA_BACKEND
    CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&resource, ID, target, cudaGraphicsRegisterFlagsNone));
#endif
    resource_ready = true;
    this->ID       = ID;
}

void Texture::Impl::deinitResource()
{
#ifdef ATCG_CUDA_BACKEND
    CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(resource));
#endif
    this->ID       = 0;
    resource_ready = false;
}

void Texture::Impl::mapResourceDevice()
{
#ifdef ATCG_CUDA_BACKEND
    if(!mapped_device)
    {
        CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &resource));
        mapped_device = true;
    }
#endif
}

void Texture::Impl::unmapResourceDevice()
{
#ifdef ATCG_CUDA_BACKEND
    if(mapped_device)
    {
        CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &resource));
    }
    mapped_device = false;
#endif
}

atcg::textureArray Texture::getTextureArray(const uint32_t mip_level) const
{
#ifdef ATCG_CUDA_BACKEND
    impl->mapResourceDevice();
    std::size_t size;
    CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&impl->dev_ptr, impl->resource, 0, mip_level));
#endif
    return impl->dev_ptr;
}

atcg::textureObject Texture::getTextureObject(const uint32_t mip_level,
                                              const glm::vec4& border_color,
                                              const bool normalized_coords,
                                              const bool normalized_float) const
{
#ifdef ATCG_CUDA_BACKEND
    if(!impl->texture_mapped)
    {
        atcg::textureArray array = getTextureArray(mip_level);

        cudaResourceDesc resDesc = {};
        resDesc.resType          = cudaResourceTypeArray;
        resDesc.res.array.array  = array;

        bool isFloat = _spec.format == TextureFormat::RFLOAT || _spec.format == TextureFormat::RGFLOAT ||
                       _spec.format == TextureFormat::RGBFLOAT || _spec.format == TextureFormat::RGBAFLOAT;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0]  = detail::toCUDAAddressMode(_spec.sampler.wrap_mode);
        texDesc.addressMode[1]  = detail::toCUDAAddressMode(_spec.sampler.wrap_mode);
        texDesc.addressMode[2]  = detail::toCUDAAddressMode(_spec.sampler.wrap_mode);
        texDesc.filterMode      = detail::toCUDATextureMode(_spec.sampler.filter_mode);
        texDesc.readMode        = isFloat            ? cudaReadModeElementType
                                  : normalized_float ? cudaReadModeNormalizedFloat
                                                     : cudaReadModeElementType;
        texDesc.borderColor[0]  = border_color.x;
        texDesc.borderColor[1]  = border_color.y;
        texDesc.borderColor[2]  = border_color.z;
        texDesc.borderColor[3]  = border_color.w;

        texDesc.normalizedCoords = normalized_coords;

        CUDA_SAFE_CALL(cudaCreateTextureObject(&impl->texture_object, &resDesc, &texDesc, NULL));
    }
#endif
    impl->texture_mapped = true;
    return impl->texture_object;
}

atcg::surfaceObject Texture::getSurfaceObject(const uint32_t mip_level) const
{
#ifdef ATCG_CUDA_BACKEND
    if(!impl->surface_mapped)
    {
        atcg::textureArray array = getTextureArray(mip_level);

        cudaResourceDesc resDesc = {};
        resDesc.resType          = cudaResourceTypeArray;
        resDesc.res.array.array  = array;

        CUDA_SAFE_CALL(cudaCreateSurfaceObject(&impl->texture_object, &resDesc));
    }
#endif
    impl->surface_mapped = true;
    return impl->surface_object;
}

bool Texture::isDeviceMapped() const
{
    return impl->mapped_device;
}

void Texture::unmapDevicePointers() const
{
#if ATCG_CUDA_BACKEND
    if(impl->surface_mapped)
    {
        CUDA_SAFE_CALL(cudaDestroySurfaceObject(impl->surface_object));
        impl->surface_mapped = false;
    }
    if(impl->texture_mapped)
    {
        CUDA_SAFE_CALL(cudaDestroyTextureObject(impl->texture_object));
        impl->texture_mapped = false;
    }
#endif
    impl->unmapResourceDevice();
}

void Texture::unmapPointers() const
{
    unmapDevicePointers();
}

void Texture::fill(void* value)
{
    glClearTexImage(_ID, 0, detail::toGLformat(_spec.format), detail::toGLtype(_spec.format), value);
}

void Texture::useForCompute(const uint32_t& slot) const
{
    unmapPointers();
    glBindImageTexture(slot, _ID, 0, GL_TRUE, 0, GL_WRITE_ONLY, detail::to2GLinternalFormat(_spec.format));
}

uint32_t Texture::channels() const
{
    return detail::num_channels(_spec.format);
}

bool Texture::isHDR() const
{
    return _spec.format == TextureFormat::RFLOAT || _spec.format == TextureFormat::RGBAFLOAT ||
           _spec.format == TextureFormat::RGBFLOAT || _spec.format == TextureFormat::RGFLOAT;
}

atcg::ref_ptr<Texture2D> Texture2D::create(const TextureSpecification& spec)
{
    return create(nullptr, spec);
}

atcg::ref_ptr<Texture2D> Texture2D::create(const void* data, const TextureSpecification& spec)
{
    atcg::ref_ptr<Texture2D> result = atcg::make_ref<Texture2D>();
    result->_spec                   = spec;
    result->_spec.width             = std::max<uint32_t>(1, result->_spec.width);
    result->_spec.height            = std::max<uint32_t>(1, result->_spec.height);

    glGenTextures(1, &(result->_ID));
    glBindTexture(GL_TEXTURE_2D, result->_ID);

    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 detail::to2GLinternalFormat(result->_spec.format),
                 result->_spec.width,
                 result->_spec.height,
                 0,
                 detail::toGLformat(result->_spec.format),
                 detail::toGLtype(result->_spec.format),
                 (void*)data);
    ATCG_LOG_ALLOCATION("Allocated Texture of size {} x {}", result->_spec.width, result->_spec.height);

    auto filtermode = detail::toGLFilterMode(result->_spec.sampler.filter_mode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, detail::toGLWrapMode(result->_spec.sampler.wrap_mode));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, detail::toGLWrapMode(result->_spec.sampler.wrap_mode));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filtermode);
    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_MAG_FILTER,
                    filtermode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : filtermode);

    if(result->_spec.sampler.mip_map)
    {
        glGenerateMipmap(GL_TEXTURE_2D);
    }

    if(result->_spec.format != TextureFormat::DEPTH) result->impl->initResource(result->_ID, GL_TEXTURE_2D);

    return result;
}

atcg::ref_ptr<Texture2D> Texture2D::create(const atcg::ref_ptr<Image> img, const TextureSpecification& spec)
{
    return create(img->data(), spec);
}

atcg::ref_ptr<Texture2D> Texture2D::create(const atcg::ref_ptr<Image> img)
{
    return create(img->data());
}

atcg::ref_ptr<Texture2D> Texture2D::create(const torch::Tensor& img)
{
    TextureSpecification spec;
    spec.width  = std::max<uint32_t>(1, img.size(1));
    spec.height = std::max<uint32_t>(1, img.size(0));
    switch(img.size(2))
    {
        case 1:
        {
            spec.format = (img.dtype() == torch::kFloat32
                               ? TextureFormat::RFLOAT
                               : (img.dtype() == torch::kInt32 ? TextureFormat::RINT : TextureFormat::RINT8));
        }
        break;
        case 2:
        {
            spec.format = (img.dtype() == torch::kFloat32 ? TextureFormat::RGFLOAT : TextureFormat::RG);
        }
        break;
        case 3:
        {
            spec.format = (img.dtype() == torch::kFloat32 ? TextureFormat::RGBFLOAT : TextureFormat::RGB);
        }
        break;
        case 4:
        {
            spec.format = (img.dtype() == torch::kFloat32 ? TextureFormat::RGBAFLOAT : TextureFormat::RGBA);
        }
        break;
    }
    return create(img.data_ptr(), spec);
}

atcg::ref_ptr<Texture2D> Texture2D::create(const torch::Tensor& img, const TextureSpecification& spec)
{
    return create(img.data_ptr(), spec);
}

Texture2D::~Texture2D()
{
    unmapPointers();
    glDeleteTextures(1, &_ID);
}

void Texture2D::setData(const torch::Tensor& data)
{
    TORCH_CHECK_GE(data.ndimension(), 2);
    TORCH_CHECK_LE(data.ndimension(), 3);
    if(data.ndimension() < 3)
    {
        data.unsqueeze_(-1);
    }

    TORCH_CHECK_EQ(data.size(0), _spec.height);
    TORCH_CHECK_EQ(data.size(1), _spec.width);
    int num_channels = detail::num_channels(_spec.format);
    TORCH_CHECK_EQ(data.size(2), num_channels);
    TORCH_CHECK_EQ(data.numel() * data.element_size(), _spec.width * _spec.height * detail::toSize(_spec.format));

    torch::Tensor pixel_data = data;
    bool cuda_copy_possible  = num_channels == 1 || num_channels == 4;
    if(data.is_cuda())
    {
        if(cuda_copy_possible)
        {
#ifdef ATCG_CUDA_BACKEND
            atcg::textureArray array = getTextureArray();
            CUDA_SAFE_CALL(cudaMemcpy2DToArray(array,
                                               0,
                                               0,
                                               pixel_data.data_ptr(),
                                               pixel_data.size(1) * pixel_data.size(2) * pixel_data.element_size(),
                                               pixel_data.size(1) * pixel_data.size(2) * pixel_data.element_size(),
                                               _spec.height,
                                               cudaMemcpyDeviceToDevice));
            unmapPointers();
#endif
            return;
        }
        ATCG_WARN("Can not copy texture data via device copy. Fall back to host-device copy");
        pixel_data = data.to(atcg::CPU);
    }

    unmapPointers();
    glBindTexture(GL_TEXTURE_2D, _ID);

    glTexSubImage2D(GL_TEXTURE_2D,
                    0,
                    0,
                    0,
                    _spec.width,
                    _spec.height,
                    detail::toGLformat(_spec.format),
                    detail::toGLtype(_spec.format),
                    (void*)pixel_data.data_ptr());
}

torch::Tensor Texture2D::getData(const torch::Device& device, const uint32_t mip_level) const
{
    int num_channels = detail::num_channels(_spec.format);
    bool hdr         = isHDR();
    int channel_size = detail::toChannelSize(_spec.format);

    torch::Tensor result;
    int height = glm::floor(_spec.height / (1 << mip_level));
    int width  = glm::floor(_spec.width / (1 << mip_level));
#ifdef ATCG_CUDA_BACKEND
    bool cuda_copy_possible = num_channels == 1 || num_channels == 4;
    if(cuda_copy_possible && device.is_cuda())
    {
        result = torch::empty({height, width, num_channels},
                              hdr ? atcg::TensorOptions::floatDeviceOptions()
                                  : (channel_size == 1 ? atcg::TensorOptions::uint8DeviceOptions()
                                                       : atcg::TensorOptions::int32DeviceOptions()));

        atcg::textureArray array = getTextureArray(mip_level);

        CUDA_SAFE_CALL(cudaMemcpy2DFromArray(result.data_ptr(),
                                             result.size(1) * result.size(2) * result.element_size(),
                                             array,
                                             0,
                                             0,
                                             result.size(1) * result.size(2) * result.element_size(),
                                             height,
                                             cudaMemcpyDeviceToDevice));

        unmapPointers();

        return result;
    }
    else if(device.is_cuda())
    {
        ATCG_WARN("Can not copy texture data via device copy. Fall back to host-device copy");
    }
#endif

    unmapPointers();

    result = torch::empty(
        {height, width, num_channels},
        hdr ? atcg::TensorOptions::floatHostOptions()
            : (channel_size == 1 ? atcg::TensorOptions::uint8HostOptions() : atcg::TensorOptions::int32HostOptions()));

    use();
    glGetTexImage(GL_TEXTURE_2D,
                  mip_level,
                  detail::toGLformat(_spec.format),
                  detail::toGLtype(_spec.format),
                  result.data_ptr());
    return result.to(device);
}

void Texture2D::use(const uint32_t& slot) const
{
    unmapPointers();
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_2D, _ID);
}

void Texture2D::generateMipmaps()
{
    use();
    _spec.sampler.mip_map = true;
    auto filtermode       = detail::toGLFilterMode(_spec.sampler.filter_mode);
    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_MAG_FILTER,
                    filtermode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : filtermode);
    glGenerateMipmap(GL_TEXTURE_2D);
}

atcg::ref_ptr<Texture> Texture2D::clone() const
{
    auto result = atcg::Texture2D::create(_spec);

    use();

    int max_level = _spec.sampler.mip_map
                        ? 1 + glm::floor(glm::log2((float)glm::max(_spec.width, glm::max(_spec.height, _spec.depth))))
                        : 1;
    for(int lvl = 0; lvl < max_level; lvl++)
    {
        int width, height;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, lvl, GL_TEXTURE_WIDTH, &width);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, lvl, GL_TEXTURE_HEIGHT, &height);

        glCopyImageSubData(_ID,
                           GL_TEXTURE_2D,
                           lvl,
                           0,
                           0,
                           0,
                           result->getID(),
                           GL_TEXTURE_2D,
                           lvl,
                           0,
                           0,
                           0,
                           width,
                           height,
                           1);
    }


    return result;
}

atcg::ref_ptr<Texture3D> Texture3D::create(const TextureSpecification& spec)
{
    return create(nullptr, spec);
}

atcg::ref_ptr<Texture3D> Texture3D::create(const void* data, const TextureSpecification& spec)
{
    atcg::ref_ptr<Texture3D> result = atcg::make_ref<Texture3D>();
    result->_spec                   = spec;
    result->_spec.width             = std::max<uint32_t>(1, result->_spec.width);
    result->_spec.height            = std::max<uint32_t>(1, result->_spec.height);
    result->_spec.depth             = std::max<uint32_t>(1, result->_spec.depth);

    glGenTextures(1, &(result->_ID));
    glBindTexture(GL_TEXTURE_3D, result->_ID);

    glTexImage3D(GL_TEXTURE_3D,
                 0,
                 detail::to2GLinternalFormat(spec.format),
                 result->_spec.width,
                 result->_spec.height,
                 result->_spec.depth,
                 0,
                 detail::toGLformat(result->_spec.format),
                 detail::toGLtype(result->_spec.format),
                 (void*)data);
    ATCG_LOG_ALLOCATION("Allocated Texture of size {} x {} x {}",
                        result->_spec.width,
                        result->_spec.height,
                        result->_spec.depth);

    auto filtermode = detail::toGLFilterMode(result->_spec.sampler.filter_mode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, detail::toGLWrapMode(result->_spec.sampler.wrap_mode));
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, detail::toGLWrapMode(result->_spec.sampler.wrap_mode));
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, detail::toGLWrapMode(result->_spec.sampler.wrap_mode));
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, filtermode);
    glTexParameteri(GL_TEXTURE_3D,
                    GL_TEXTURE_MAG_FILTER,
                    filtermode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : filtermode);

    if(result->_spec.sampler.mip_map)
    {
        glGenerateMipmap(GL_TEXTURE_3D);
    }

    if(result->_spec.format != TextureFormat::DEPTH) result->impl->initResource(result->_ID, GL_TEXTURE_3D);

    return result;
}

atcg::ref_ptr<Texture3D> Texture3D::create(const torch::Tensor& img)
{
    TextureSpecification spec;
    spec.width  = std::max<uint32_t>(1, img.size(2));
    spec.height = std::max<uint32_t>(1, img.size(1));
    spec.depth  = std::max<uint32_t>(1, img.size(0));
    switch(img.size(3))
    {
        case 1:
        {
            spec.format = (img.dtype() == torch::kFloat32
                               ? TextureFormat::RFLOAT
                               : (img.dtype() == torch::kInt32 ? TextureFormat::RINT : TextureFormat::RINT8));
        }
        break;
        case 2:
        {
            spec.format = (img.dtype() == torch::kFloat32 ? TextureFormat::RGFLOAT : TextureFormat::RG);
        }
        break;
        case 3:
        {
            spec.format = (img.dtype() == torch::kFloat32 ? TextureFormat::RGBFLOAT : TextureFormat::RGB);
        }
        break;
        case 4:
        {
            spec.format = (img.dtype() == torch::kFloat32 ? TextureFormat::RGBAFLOAT : TextureFormat::RGBA);
        }
        break;
    }
    return create(img.data_ptr(), spec);
}

atcg::ref_ptr<Texture3D> Texture3D::create(const torch::Tensor& img, const TextureSpecification& spec)
{
    return create(img.data_ptr(), spec);
}

Texture3D::~Texture3D()
{
    unmapPointers();
    glDeleteTextures(1, &_ID);
}

void Texture3D::setData(const torch::Tensor& data)
{
    TORCH_CHECK_GE(data.ndimension(), 3);
    TORCH_CHECK_LE(data.ndimension(), 4);
    if(data.ndimension() < 4)
    {
        data.unsqueeze_(-1);
    }

    TORCH_CHECK_EQ(data.size(0), _spec.depth);
    TORCH_CHECK_EQ(data.size(1), _spec.height);
    TORCH_CHECK_EQ(data.size(2), _spec.width);
    int num_channels = detail::num_channels(_spec.format);
    TORCH_CHECK_EQ(data.size(3), num_channels);
    TORCH_CHECK_EQ(data.numel() * data.element_size(),
                   _spec.depth * _spec.width * _spec.height * detail::toSize(_spec.format));

    torch::Tensor pixel_data = data;
    bool cuda_copy_possible  = num_channels == 1 || num_channels == 4;
    if(data.is_cuda())
    {
        if(cuda_copy_possible)
        {
#ifdef ATCG_CUDA_BACKEND
            atcg::textureArray array   = getTextureArray();
            cudaChannelFormatDesc desc = {};
            cudaExtent ext             = {};
            unsigned int array_flags   = 0;

            CUDA_SAFE_CALL(cudaArrayGetInfo(&desc, &ext, &array_flags, array));

            cudaMemcpy3DParms p = {0};
            p.dstArray          = array;
            p.kind              = cudaMemcpyDeviceToDevice;
            p.srcPtr.ptr        = pixel_data.contiguous().data_ptr();
            p.srcPtr.pitch      = ext.width * num_channels;
            p.srcPtr.xsize      = ext.width;
            p.srcPtr.ysize      = ext.height;
            p.extent            = ext;

            CUDA_SAFE_CALL(cudaMemcpy3D(&p));
            unmapPointers();
#endif
            return;
        }
        ATCG_WARN("Can not copy texture data via device copy. Fall back to host-device copy");
        pixel_data = data.to(atcg::CPU);
    }

    unmapPointers();
    glBindTexture(GL_TEXTURE_3D, _ID);

    glTexSubImage3D(GL_TEXTURE_3D,
                    0,
                    0,
                    0,
                    0,
                    _spec.width,
                    _spec.height,
                    _spec.depth,
                    detail::toGLformat(_spec.format),
                    detail::toGLtype(_spec.format),
                    (void*)pixel_data.data_ptr());
}

torch::Tensor Texture3D::getData(const torch::Device& device, const uint32_t mip_level) const
{
    int num_channels = detail::num_channels(_spec.format);
    bool hdr         = isHDR();
    int channel_size = detail::toChannelSize(_spec.format);

    torch::Tensor result;
    int depth  = glm::floor(_spec.depth / (1 << mip_level));
    int height = glm::floor(_spec.height / (1 << mip_level));
    int width  = glm::floor(_spec.width / (1 << mip_level));
#ifdef ATCG_CUDA_BACKEND
    bool cuda_copy_possible = num_channels == 1 || num_channels == 4;
    if(cuda_copy_possible && device.is_cuda())
    {
        result = torch::empty({depth, height, width, num_channels},
                              hdr ? atcg::TensorOptions::floatDeviceOptions()
                                  : (channel_size == 1 ? atcg::TensorOptions::uint8DeviceOptions()
                                                       : atcg::TensorOptions::int32DeviceOptions()));

        atcg::textureArray array = getTextureArray(mip_level);

        cudaChannelFormatDesc desc = {};
        cudaExtent ext             = {};
        unsigned int array_flags   = 0;

        CUDA_SAFE_CALL(cudaArrayGetInfo(&desc, &ext, &array_flags, array));

        cudaMemcpy3DParms p = {0};
        p.srcArray          = array;
        p.kind              = cudaMemcpyDeviceToDevice;
        p.dstPtr.ptr        = result.contiguous().data_ptr();
        p.dstPtr.pitch      = ext.width * num_channels;
        p.dstPtr.xsize      = ext.width;
        p.dstPtr.ysize      = ext.height;
        p.extent            = ext;

        CUDA_SAFE_CALL(cudaMemcpy3D(&p));

        unmapPointers();

        return result;
    }
    else if(device.is_cuda())
    {
        ATCG_WARN("Can not copy texture data via device copy. Fall back to host-device copy");
    }
#endif

    unmapPointers();

    result = torch::empty(
        {depth, height, width, num_channels},
        hdr ? atcg::TensorOptions::floatHostOptions()
            : (channel_size == 1 ? atcg::TensorOptions::uint8HostOptions() : atcg::TensorOptions::int32HostOptions()));

    use();
    glGetTexImage(GL_TEXTURE_3D,
                  mip_level,
                  detail::toGLformat(_spec.format),
                  detail::toGLtype(_spec.format),
                  result.data_ptr());
    return result.to(device);
}

void Texture3D::use(const uint32_t& slot) const
{
    unmapPointers();
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_3D, _ID);
}

void Texture3D::generateMipmaps()
{
    use();
    _spec.sampler.mip_map = true;
    auto filtermode       = detail::toGLFilterMode(_spec.sampler.filter_mode);
    glTexParameteri(GL_TEXTURE_3D,
                    GL_TEXTURE_MAG_FILTER,
                    filtermode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : filtermode);
    glGenerateMipmap(GL_TEXTURE_3D);
}

atcg::ref_ptr<Texture> Texture3D::clone() const
{
    auto result = atcg::Texture2D::create(_spec);

    use();

    int max_level = _spec.sampler.mip_map
                        ? 1 + glm::floor(glm::log2((float)glm::max(_spec.width, glm::max(_spec.height, _spec.depth))))
                        : 1;
    for(int lvl = 0; lvl < max_level + 1; lvl++)
    {
        int width, height, depth;
        glGetTexLevelParameteriv(GL_TEXTURE_3D, lvl, GL_TEXTURE_WIDTH, &width);
        glGetTexLevelParameteriv(GL_TEXTURE_3D, lvl, GL_TEXTURE_HEIGHT, &height);
        glGetTexLevelParameteriv(GL_TEXTURE_3D, lvl, GL_TEXTURE_DEPTH, &depth);

        glCopyImageSubData(_ID,
                           GL_TEXTURE_3D,
                           lvl,
                           0,
                           0,
                           0,
                           result->getID(),
                           GL_TEXTURE_3D,
                           lvl,
                           0,
                           0,
                           0,
                           width,
                           height,
                           depth);
    }


    return result;
}

atcg::ref_ptr<TextureCube> TextureCube::create(const TextureSpecification& spec)
{
    atcg::ref_ptr<TextureCube> result = atcg::make_ref<TextureCube>();
    result->_spec                     = spec;

    glGenTextures(1, &(result->_ID));
    glBindTexture(GL_TEXTURE_CUBE_MAP, result->_ID);

    for(uint32_t i = 0; i < 6; ++i)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                     0,
                     detail::to2GLinternalFormat(spec.format),
                     spec.width,
                     spec.height,
                     0,
                     detail::toGLformat(spec.format),
                     detail::toGLtype(spec.format),
                     nullptr);
    }

    auto filtermode = detail::toGLFilterMode(spec.sampler.filter_mode);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, detail::toGLWrapMode(spec.sampler.wrap_mode));
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, detail::toGLWrapMode(spec.sampler.wrap_mode));
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, detail::toGLWrapMode(spec.sampler.wrap_mode));
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, filtermode);
    glTexParameteri(GL_TEXTURE_CUBE_MAP,
                    GL_TEXTURE_MAG_FILTER,
                    filtermode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : filtermode);

    if(spec.sampler.mip_map)
    {
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    }

    return result;
}

TextureCube::~TextureCube()
{
    glDeleteTextures(1, &_ID);
}

void TextureCube::use(const uint32_t& slot) const
{
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_CUBE_MAP, _ID);
}

void TextureCube::setData(const torch::Tensor& data)
{
    TORCH_CHECK_EQ(data.numel() * data.element_size() / 6, _spec.width * _spec.height * detail::toSize(_spec.format));
    TORCH_CHECK_EQ(data.size(0), 6);
    TORCH_CHECK_EQ(data.size(1), _spec.height);
    TORCH_CHECK_EQ(data.size(2), _spec.width);
    int num_channels = detail::num_channels(_spec.format);
    TORCH_CHECK_EQ(data.size(3), num_channels);

    torch::Tensor pixel_data = data.to(atcg::CPU);

    unmapPointers();
    glBindTexture(GL_TEXTURE_CUBE_MAP, _ID);

    for(int i = 0; i < 6; ++i)
    {
        glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                        0,
                        0,
                        0,
                        _spec.width,
                        _spec.height,
                        detail::toGLformat(_spec.format),
                        detail::toGLtype(_spec.format),
                        (void*)pixel_data.index({i, 0, 0, 0}).data_ptr());
    }
}

torch::Tensor TextureCube::getData(const torch::Device& device, const uint32_t mip_level) const
{
    int num_channels = detail::num_channels(_spec.format);
    bool hdr         = _spec.format == TextureFormat::RFLOAT || _spec.format == TextureFormat::RGBAFLOAT ||
               _spec.format == TextureFormat::RGBFLOAT;
    int channel_size = detail::toChannelSize(_spec.format);

    auto result = torch::empty(
        {6, _spec.height, _spec.width, num_channels},
        hdr ? atcg::TensorOptions::floatHostOptions()
            : (num_channels == 1 ? atcg::TensorOptions::uint8HostOptions() : atcg::TensorOptions::int32HostOptions()));

    unmapPointers();
    glBindTexture(GL_TEXTURE_CUBE_MAP, _ID);

    for(int i = 0; i < 6; ++i)
    {
        glGetTexImage(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                      mip_level,
                      detail::toGLformat(_spec.format),
                      detail::toGLtype(_spec.format),
                      (void*)result.index({i, 0, 0, 0}).data_ptr());
    }

    return result.to(device);
}

void TextureCube::generateMipmaps()
{
    use();
    _spec.sampler.mip_map = true;
    auto filtermode       = detail::toGLFilterMode(_spec.sampler.filter_mode);
    glTexParameteri(GL_TEXTURE_CUBE_MAP,
                    GL_TEXTURE_MAG_FILTER,
                    filtermode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : filtermode);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
}

atcg::ref_ptr<Texture> TextureCube::clone() const
{
    auto result = atcg::TextureCube::create(_spec);

    use();

    int max_level = _spec.sampler.mip_map
                        ? 1 + glm::floor(glm::log2((float)glm::max(_spec.width, glm::max(_spec.height, _spec.depth))))
                        : 1;
    for(int i = 0; i < 6; ++i)
    {
        for(int lvl = 0; lvl < max_level + 1; lvl++)
        {
            int width, height;
            glGetTexLevelParameteriv(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, lvl, GL_TEXTURE_WIDTH, &width);
            glGetTexLevelParameteriv(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, lvl, GL_TEXTURE_HEIGHT, &height);

            glCopyImageSubData(_ID,
                               GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                               lvl,
                               0,
                               0,
                               0,
                               result->getID(),
                               GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                               lvl,
                               0,
                               0,
                               0,
                               width,
                               height,
                               1);
        }
    }


    return result;
}


atcg::ref_ptr<TextureArray> TextureArray::create(const TextureSpecification& spec)
{
    return create(nullptr, spec);
}

atcg::ref_ptr<TextureArray> TextureArray::create(const void* data, const TextureSpecification& spec)
{
    atcg::ref_ptr<TextureArray> result = atcg::make_ref<TextureArray>();
    result->_spec                      = spec;
    result->_spec.width                = std::max<uint32_t>(1, result->_spec.width);
    result->_spec.height               = std::max<uint32_t>(1, result->_spec.height);
    result->_spec.depth                = std::max<uint32_t>(1, result->_spec.depth);

    glGenTextures(1, &(result->_ID));
    glBindTexture(GL_TEXTURE_2D_ARRAY, result->_ID);

    glTexImage3D(GL_TEXTURE_2D_ARRAY,
                 0,
                 detail::to2GLinternalFormat(spec.format),
                 result->_spec.width,
                 result->_spec.height,
                 result->_spec.depth,
                 0,
                 detail::toGLformat(result->_spec.format),
                 detail::toGLtype(result->_spec.format),
                 (void*)data);
    ATCG_LOG_ALLOCATION("Allocated Texture of size {} x {} x {}",
                        result->_spec.width,
                        result->_spec.height,
                        result->_spec.depth);

    auto filtermode = detail::toGLFilterMode(result->_spec.sampler.filter_mode);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, detail::toGLWrapMode(result->_spec.sampler.wrap_mode));
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, detail::toGLWrapMode(result->_spec.sampler.wrap_mode));
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_R, detail::toGLWrapMode(result->_spec.sampler.wrap_mode));
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, filtermode);
    glTexParameteri(GL_TEXTURE_2D_ARRAY,
                    GL_TEXTURE_MAG_FILTER,
                    filtermode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : filtermode);

    if(result->_spec.sampler.mip_map)
    {
        glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
    }

    if(result->_spec.format != TextureFormat::DEPTH) result->impl->initResource(result->_ID, GL_TEXTURE_2D_ARRAY);

    return result;
}

atcg::ref_ptr<TextureArray> TextureArray::create(const torch::Tensor& img)
{
    TextureSpecification spec;
    spec.width  = std::max<uint32_t>(1, img.size(2));
    spec.height = std::max<uint32_t>(1, img.size(1));
    spec.depth  = std::max<uint32_t>(1, img.size(0));
    switch(img.size(3))
    {
        case 1:
        {
            spec.format = (img.dtype() == torch::kFloat32 ? TextureFormat::RFLOAT : TextureFormat::RINT8);
        }
        break;
        case 2:
        {
            spec.format = (img.dtype() == torch::kFloat32 ? TextureFormat::RGFLOAT : TextureFormat::RG);
        }
        break;
        case 3:
        {
            spec.format = (img.dtype() == torch::kFloat32 ? TextureFormat::RGBFLOAT : TextureFormat::RGB);
        }
        break;
        case 4:
        {
            spec.format = (img.dtype() == torch::kFloat32 ? TextureFormat::RGBAFLOAT : TextureFormat::RGBA);
        }
        break;
    }
    return create(img.data_ptr(), spec);
}

atcg::ref_ptr<TextureArray> TextureArray::create(const torch::Tensor& img, const TextureSpecification& spec)
{
    return create(img.data_ptr(), spec);
}

TextureArray::~TextureArray()
{
    unmapPointers();
    glDeleteTextures(1, &_ID);
}

void TextureArray::setData(const torch::Tensor& data)
{
    TORCH_CHECK_GE(data.ndimension(), 3);
    TORCH_CHECK_LE(data.ndimension(), 4);
    if(data.ndimension() < 4)
    {
        data.unsqueeze_(-1);
    }

    TORCH_CHECK_EQ(data.size(0), _spec.depth);
    TORCH_CHECK_EQ(data.size(1), _spec.height);
    TORCH_CHECK_EQ(data.size(2), _spec.width);
    int num_channels = detail::num_channels(_spec.format);
    TORCH_CHECK_EQ(data.size(3), num_channels);
    TORCH_CHECK_EQ(data.numel() * data.element_size(),
                   _spec.depth * _spec.width * _spec.height * detail::toSize(_spec.format));

    torch::Tensor pixel_data = data;
    bool cuda_copy_possible  = num_channels == 1 || num_channels == 4;
    if(data.is_cuda())
    {
        if(cuda_copy_possible)
        {
#ifdef ATCG_CUDA_BACKEND
            atcg::textureArray array   = getTextureArray();
            cudaChannelFormatDesc desc = {};
            cudaExtent ext             = {};
            unsigned int array_flags   = 0;

            CUDA_SAFE_CALL(cudaArrayGetInfo(&desc, &ext, &array_flags, array));

            cudaMemcpy3DParms p = {0};
            p.dstArray          = array;
            p.kind              = cudaMemcpyDeviceToDevice;
            p.srcPtr.ptr        = pixel_data.contiguous().data_ptr();
            p.srcPtr.pitch      = ext.width * num_channels;
            p.srcPtr.xsize      = ext.width;
            p.srcPtr.ysize      = ext.height;
            p.extent            = ext;

            CUDA_SAFE_CALL(cudaMemcpy3D(&p));
            unmapPointers();
#endif
            return;
        }
        ATCG_WARN("Can not copy texture data via device copy. Fall back to host-device copy");
        pixel_data = data.to(atcg::CPU);
    }

    unmapPointers();
    glBindTexture(GL_TEXTURE_2D_ARRAY, _ID);

    glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
                    0,
                    0,
                    0,
                    0,
                    _spec.width,
                    _spec.height,
                    _spec.depth,
                    detail::toGLformat(_spec.format),
                    detail::toGLtype(_spec.format),
                    (void*)pixel_data.data_ptr());
}

torch::Tensor TextureArray::getData(const torch::Device& device, const uint32_t mip_level) const
{
    int num_channels = detail::num_channels(_spec.format);
    bool hdr         = _spec.format == TextureFormat::RFLOAT || _spec.format == TextureFormat::RGBAFLOAT ||
               _spec.format == TextureFormat::RGBFLOAT;
    int channel_size = detail::toChannelSize(_spec.format);

    torch::Tensor result;
    int depth  = glm::floor(_spec.depth / (1 << mip_level));
    int height = glm::floor(_spec.height / (1 << mip_level));
    int width  = glm::floor(_spec.width / (1 << mip_level));
#ifdef ATCG_CUDA_BACKEND
    bool cuda_copy_possible = num_channels == 1 || num_channels == 4;
    if(cuda_copy_possible && device.is_cuda())
    {
        result = torch::empty({height, width, num_channels},
                              hdr ? atcg::TensorOptions::floatDeviceOptions()
                                  : (channel_size == 1 ? atcg::TensorOptions::uint8DeviceOptions()
                                                       : atcg::TensorOptions::int32DeviceOptions()));

        atcg::textureArray array = getTextureArray(mip_level);

        cudaChannelFormatDesc desc = {};
        cudaExtent ext             = {};
        unsigned int array_flags   = 0;

        CUDA_SAFE_CALL(cudaArrayGetInfo(&desc, &ext, &array_flags, array));

        cudaMemcpy3DParms p = {0};
        p.srcArray          = array;
        p.kind              = cudaMemcpyDeviceToDevice;
        p.dstPtr.ptr        = result.contiguous().data_ptr();
        p.dstPtr.pitch      = ext.width * num_channels;
        p.dstPtr.xsize      = ext.width;
        p.dstPtr.ysize      = ext.height;
        p.extent            = ext;

        CUDA_SAFE_CALL(cudaMemcpy3D(&p));

        unmapPointers();

        return result;
    }
    else if(device.is_cuda())
    {
        ATCG_WARN("Can not copy texture data via device copy. Fall back to host-device copy");
    }
#endif

    unmapPointers();

    result = torch::empty(
        {depth, height, width, num_channels},
        hdr ? atcg::TensorOptions::floatHostOptions()
            : (channel_size == 1 ? atcg::TensorOptions::uint8HostOptions() : atcg::TensorOptions::int32HostOptions()));

    use();
    glGetTexImage(GL_TEXTURE_2D_ARRAY,
                  mip_level,
                  detail::toGLformat(_spec.format),
                  detail::toGLtype(_spec.format),
                  result.data_ptr());
    return result.to(device);
}

void TextureArray::use(const uint32_t& slot) const
{
    unmapPointers();
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_2D_ARRAY, _ID);
}

void TextureArray::generateMipmaps()
{
    use();
    _spec.sampler.mip_map = true;
    auto filtermode       = detail::toGLFilterMode(_spec.sampler.filter_mode);
    glTexParameteri(GL_TEXTURE_2D_ARRAY,
                    GL_TEXTURE_MAG_FILTER,
                    filtermode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : filtermode);
    glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
}

atcg::ref_ptr<Texture> TextureArray::clone() const
{
    auto result = atcg::TextureArray::create(_spec);

    use();

    int max_level = _spec.sampler.mip_map
                        ? 1 + glm::floor(glm::log2((float)glm::max(_spec.width, glm::max(_spec.height, _spec.depth))))
                        : 1;
    for(int lvl = 0; lvl < max_level + 1; lvl++)
    {
        int width, height;
        glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, lvl, GL_TEXTURE_WIDTH, &width);
        glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, lvl, GL_TEXTURE_HEIGHT, &height);

        glCopyImageSubData(_ID,
                           GL_TEXTURE_2D_ARRAY,
                           lvl,
                           0,
                           0,
                           0,
                           result->getID(),
                           GL_TEXTURE_2D_ARRAY,
                           lvl,
                           0,
                           0,
                           0,
                           width,
                           height,
                           _spec.depth);
    }


    return result;
}

}    // namespace atcg