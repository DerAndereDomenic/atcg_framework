#pragma once

#include <CuDiff/ext/glm.h>
#include <Core/GlobalAtomicAdd.h>

namespace atcg
{

template<typename T>
template<typename iuv_t>
ATCG_INLINE ATCG_HOST_DEVICE std::byte* TextureInterface<T>::getTexelPtr(const iuv_t& texel) const
{
    size_t index = toIndex(texel);
    if(_spec.isFloat() || _spec.isInt())
    {
        float* pixels = reinterpret_cast<float*>(_data);
        return (std::byte*)&pixels[index];
    }
    else
    {
        return &_data[index];
    }
}

template<typename T>
ATCG_INLINE ATCG_HOST_DEVICE T TextureInterface<T>::convertData(const std::byte* data) const
{
    if(_spec.isFloat() || _spec.isInt())
    {
        const float* pixels = reinterpret_cast<const float*>(data);
        if constexpr((std::is_same_v<T, float>) || (std::is_same_v<T, int32_t>))
        {
            return (T)pixels[0];
        }
        else
        {
            T result(0);
            for(uint32_t i = 0; i < vec_traits<T>::dim; ++i)
            {
                result[i] = pixels[i];
            }
            return result;
        }
    }
    else
    {
        if constexpr(std::is_same_v<T, float>)
        {
            return (float)data[0] / 255.0f;
        }
        else
        {
            T result(0);
            for(uint32_t i = 0; i < vec_traits<T>::dim; ++i)
            {
                result[i] = (float)data[i] / 255.0f;
            }
            return result;
        }
    }
}

template<typename T>
template<typename iuv_t>
ATCG_INLINE ATCG_HOST_DEVICE T TextureInterface<T>::fetchTexel(const iuv_t& texel) const
{
    std::byte* data = getTexelPtr(texel);
    return convertData(data);
}

template<typename T>
template<typename iuv_t, TexelWriteMode write_mode>
ATCG_INLINE ATCG_HOST_DEVICE void TextureInterface<T>::writeTexel(const T& val, const iuv_t& texel)
{
    std::byte* data = getTexelPtr(texel);
    TexelUpdater<write_mode> updater;
    if(_spec.isFloat() || _spec.isInt())
    {
        float* pixels = reinterpret_cast<float*>(data);
        if constexpr((std::is_same_v<T, float>) || (std::is_same_v<T, int32_t>))
        {
            updater(pixels, val);
        }
        else
        {
            for(uint32_t i = 0; i < vec_traits<T>::dim; ++i)
            {
                updater(&pixels[i], val[i]);
            }
        }
    }
    else
    {
        printf("Atomic add not supported for normalized uint8_t textures\n");
    }
}

template<typename T>
ATCG_INLINE ATCG_HOST_DEVICE size_t TextureInterface<T>::toIndex(const glm::ivec2& texel) const
{
    if(_spec.sampler.wrap_mode == TextureWrapMode::REPEAT)
    {
        int32_t x = texel.x % (int32_t)_spec.width;
        int32_t y = texel.y % (int32_t)_spec.height;
        if(x < 0) x += _spec.width;
        if(y < 0) y += _spec.height;
        return size_t((y * _spec.width + x) * _spec.numChannels());
    }
    else /*if(_spec.sampler.wrap_mode == TextureWrapMode::CLAMP_TO_EDGE)*/
    {
        int32_t x = glm::clamp(texel.x, 0, (int32_t)_spec.width - 1);
        int32_t y = glm::clamp(texel.y, 0, (int32_t)_spec.height - 1);
        return size_t((y * _spec.width + x) * _spec.numChannels());
    }
}

template<typename T>
ATCG_INLINE ATCG_HOST_DEVICE size_t TextureInterface<T>::toIndex(const glm::ivec3& texel) const
{
    if(_spec.sampler.wrap_mode == TextureWrapMode::REPEAT)
    {
        int32_t x = texel.x % (int32_t)_spec.width;
        int32_t y = texel.y % (int32_t)_spec.height;
        int32_t z = texel.z % (int32_t)_spec.depth;
        if(x < 0) x += _spec.width;
        if(y < 0) y += _spec.height;
        if(z < 0) z += _spec.depth;
        return size_t((texel.x + texel.y * _spec.width + texel.z * _spec.width * _spec.height) * _spec.numChannels());
    }
    else /*if(_spec.sampler.wrap_mode == TextureWrapMode::CLAMP_TO_EDGE)*/
    {
        int32_t x = glm::clamp(texel.x, 0, (int32_t)_spec.width - 1);
        int32_t y = glm::clamp(texel.y, 0, (int32_t)_spec.height - 1);
        int32_t z = glm::clamp(texel.z, 0, (int32_t)_spec.depth - 1);
        return size_t((z * _spec.width * _spec.height + y * _spec.width + x) * _spec.numChannels());
    }
}

template<typename T>
template<typename uv_t>
ATCG_INLINE ATCG_HOST_DEVICE auto TextureSampler<T>::read(const uv_t& uv) const
{
    auto x = _read_interpolated(uv);
    return x;
}

template<typename T>
template<typename uv_t>
ATCG_HOST_DEVICE auto TextureSampler<T>::_read_interpolated(const uv_t& uv) const
{
    switch(_spec.sampler.filter_mode)
    {
        case TextureFilterMode::LINEAR:
        case TextureFilterMode::MIPMAP_LINEAR:
        {
            InterpolationReader<T, TextureFilterMode::LINEAR> reader((TextureInterface<T>*)this);
            return reader(uv);
        }
        break;
        case TextureFilterMode::NEAREST:
        default:
        {
            InterpolationReader<T, TextureFilterMode::NEAREST> reader((TextureInterface<T>*)this);
            return reader(uv);
        }
        break;
    }
}

template<typename T>
template<typename uv_t>
ATCG_HOST_DEVICE auto InterpolationReader<T, TextureFilterMode::NEAREST>::operator()(const uv_t& uv) const
{
    if constexpr(std::is_same_v<CuDiff::dual_value_type_t<uv_t>, glm::vec2>)
    {
        return _read_2d(uv);
    }
    else if constexpr(std::is_same_v<CuDiff::dual_value_type_t<uv_t>, glm::vec3>)
    {
        return _read_3d(uv);
    }
    else
    {
        if constexpr(CuDiff::is_dual_v<uv_t>)
        {
            return CuDiff::Dual<CuDiff::dual_component_count<uv_t>::num_variables, T>(T(0));
        }
        else
        {
            return T(0);
        }
    }
}

template<typename T>
template<typename uv_t>
ATCG_HOST_DEVICE auto InterpolationReader<T, TextureFilterMode::NEAREST>::_read_2d(const uv_t& uv) const
{
    auto [uv_x, uv_y] = CuDiff::unwrap(uv);

    uint32_t width   = _texture->getSpecification().width;
    uint32_t height  = _texture->getSpecification().height;
    uint32_t texel_x = (uint32_t)(uv_x * width);
    uint32_t texel_y = (uint32_t)(uv_y * height);

    T value = _texture->fetchTexel(glm::ivec2(texel_x, texel_y));

    if constexpr(CuDiff::is_dual_v<uv_t>)
    {
        return CuDiff::Dual<CuDiff::dual_component_count<uv_t>::num_variables, T>(value);
    }
    else
    {
        return value;
    }
}

template<typename T>
template<typename uv_t>
ATCG_HOST_DEVICE auto InterpolationReader<T, TextureFilterMode::NEAREST>::_read_3d(const uv_t& uv) const
{
    auto [uv_x, uv_y, uv_z] = CuDiff::unwrap(uv);

    uint32_t width   = _texture->getSpecification().width;
    uint32_t height  = _texture->getSpecification().height;
    uint32_t depth   = _texture->getSpecification().depth;
    uint32_t texel_x = (uint32_t)(uv_x * width);
    uint32_t texel_y = (uint32_t)(uv_y * height);
    uint32_t texel_z = (uint32_t)(uv_z * depth);

    T value = _texture->fetchTexel(glm::ivec3(texel_x, texel_y, texel_z));

    if constexpr(CuDiff::is_dual_v<uv_t>)
    {
        return CuDiff::Dual<CuDiff::dual_component_count<uv_t>::num_variables, T>(value);
    }
    else
    {
        return value;
    }
}

template<typename T>
template<typename uv_t>
ATCG_HOST_DEVICE auto InterpolationReader<T, TextureFilterMode::LINEAR>::operator()(const uv_t& uv) const
{
    if constexpr(std::is_same_v<CuDiff::dual_value_type_t<uv_t>, glm::vec2>)
    {
        return _read_2d(uv);
    }
    else if constexpr(std::is_same_v<CuDiff::dual_value_type_t<uv_t>, glm::vec3>)
    {
        return _read_3d(uv);
    }
    else
    {
        if constexpr(CuDiff::is_dual_v<uv_t>)
        {
            return CuDiff::Dual<CuDiff::dual_component_count<uv_t>::num_variables, T>(T(0));
        }
        else
        {
            return T(0);
        }
    }
}

template<typename T>
template<typename uv_t>
ATCG_HOST_DEVICE auto InterpolationReader<T, TextureFilterMode::LINEAR>::_read_2d(const uv_t& uv) const
{
    auto [uv_x, uv_y] = CuDiff::unwrap(uv);

    uint32_t width  = _texture->getSpecification().width;
    uint32_t height = _texture->getSpecification().height;
    auto fx         = uv_x * (width - 1);
    auto fy         = uv_y * (height - 1);

    int x0 = static_cast<int>(glm::floor(fx));
    int y0 = static_cast<int>(glm::floor(fy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    auto tx = fx - x0;
    auto ty = fy - y0;

    T c00 = _texture->fetchTexel(glm::ivec2(x0, y0));
    T c10 = _texture->fetchTexel(glm::ivec2(x1, y0));
    T c01 = _texture->fetchTexel(glm::ivec2(x0, y1));
    T c11 = _texture->fetchTexel(glm::ivec2(x1, y1));

    auto cx0 = c00 * (1.0f - tx) + tx * c10;
    auto cx1 = c01 * (1.0f - tx) + tx * c11;
    auto res = cx0 * (1.0f - ty) + ty * cx1;

    return res;
}

template<typename T>
template<typename uv_t>
ATCG_HOST_DEVICE auto InterpolationReader<T, TextureFilterMode::LINEAR>::_read_3d(const uv_t& uv) const
{
    auto [uv_x, uv_y, uv_z] = CuDiff::unwrap(uv);

    uint32_t width  = _texture->getSpecification().width;
    uint32_t height = _texture->getSpecification().height;
    uint32_t depth  = _texture->getSpecification().depth;
    auto fx         = uv_x * (width - 1);
    auto fy         = uv_y * (height - 1);
    auto fz         = uv_z * (depth - 1);

    int x0 = static_cast<int>(glm::floor(fx));
    int y0 = static_cast<int>(glm::floor(fy));
    int z0 = static_cast<int>(glm::floor(fz));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    auto tx = fx - x0;
    auto ty = fy - y0;
    auto tz = fz - z0;

    T c000 = _texture->fetchTexel(glm::ivec3(x0, y0, z0));
    T c100 = _texture->fetchTexel(glm::ivec3(x1, y0, z0));
    T c010 = _texture->fetchTexel(glm::ivec3(x0, y1, z0));
    T c110 = _texture->fetchTexel(glm::ivec3(x1, y1, z0));
    T c001 = _texture->fetchTexel(glm::ivec3(x0, y0, z1));
    T c101 = _texture->fetchTexel(glm::ivec3(x1, y0, z1));
    T c011 = _texture->fetchTexel(glm::ivec3(x0, y1, z1));
    T c111 = _texture->fetchTexel(glm::ivec3(x1, y1, z1));

    auto cx00 = c000 * (1.0f - tx) + tx * c100;
    auto cx10 = c010 * (1.0f - tx) + tx * c110;
    auto cx01 = c001 * (1.0f - tx) + tx * c101;
    auto cx11 = c011 * (1.0f - tx) + tx * c111;

    auto cxy0 = cx00 * (1.0f - ty) + ty * cx10;
    auto cxy1 = cx01 * (1.0f - ty) + ty * cx11;

    auto res = cxy0 * (1.0f - tz) + tz * cxy1;

    return res;
}

template<typename T>
template<typename uv_t, TexelWriteMode write_mode>
ATCG_HOST_DEVICE void TextureWriter<T>::write(const T& val, const uv_t& uv)
{
    switch(_spec.sampler.filter_mode)
    {
        case TextureFilterMode::LINEAR:
        case TextureFilterMode::MIPMAP_LINEAR:
        {
            InterpolationWriter<T, write_mode, TextureFilterMode::LINEAR> writer((TextureInterface<T>*)this);
            writer(val, uv);
        }
        break;
        default:
        case TextureFilterMode::NEAREST:
        {
            InterpolationWriter<T, write_mode, TextureFilterMode::NEAREST> writer((TextureInterface<T>*)this);
            writer(val, uv);
        }
        break;
    }
}

template<typename T, TexelWriteMode write_mode>
template<typename uv_t>
ATCG_HOST_DEVICE void InterpolationWriter<T, write_mode, TextureFilterMode::NEAREST>::operator()(const T& val,
                                                                                                 const uv_t& texel)
{
    if constexpr(std::is_same_v<CuDiff::dual_value_type_t<uv_t>, glm::vec2>)
    {
        _write_2d(val, texel);
    }
    else /*if constexpr(std::is_same_v<CuDiff::dual_value_type_t<uv_t>, glm::vec3>)*/
    {
        _write_3d(val, texel);
    }
}

template<typename T, TexelWriteMode write_mode>
template<typename uv_t>
ATCG_HOST_DEVICE void InterpolationWriter<T, write_mode, TextureFilterMode::NEAREST>::_write_2d(const T& val,
                                                                                                const uv_t& uv)
{
    uint32_t width   = _texture->getSpecification().width;
    uint32_t height  = _texture->getSpecification().height;
    uint32_t texel_x = (uint32_t)(uv.x * width);
    uint32_t texel_y = (uint32_t)(uv.y * height);

    _texture->writeTexel(val, glm::ivec2(texel_x, texel_y));
}

template<typename T, TexelWriteMode write_mode>
template<typename uv_t>
ATCG_HOST_DEVICE void InterpolationWriter<T, write_mode, TextureFilterMode::NEAREST>::_write_3d(const T& val,
                                                                                                const uv_t& uv)
{
    uint32_t width   = _texture->getSpecification().width;
    uint32_t height  = _texture->getSpecification().height;
    uint32_t depth   = _texture->getSpecification().depth;
    uint32_t texel_x = (uint32_t)(uv.x * width);
    uint32_t texel_y = (uint32_t)(uv.y * height);
    uint32_t texel_z = (uint32_t)(uv.z * depth);

    _texture->writeTexel(val, glm::ivec3(texel_x, texel_y, texel_z));
}

template<typename T, TexelWriteMode write_mode>
template<typename uv_t>
ATCG_HOST_DEVICE void InterpolationWriter<T, write_mode, TextureFilterMode::LINEAR>::operator()(const T& val,
                                                                                                const uv_t& texel)
{
    if constexpr(std::is_same_v<CuDiff::dual_value_type_t<uv_t>, glm::vec2>)
    {
        _write_2d(val, texel);
    }
    else /*if constexpr(std::is_same_v<CuDiff::dual_value_type_t<uv_t>, glm::vec3>)*/
    {
        _write_3d(val, texel);
    }
}


template<typename T, TexelWriteMode write_mode>
template<typename uv_t>
ATCG_HOST_DEVICE void InterpolationWriter<T, write_mode, TextureFilterMode::LINEAR>::_write_2d(const T& val,
                                                                                               const uv_t& uv)
{
    uint32_t width  = _texture->getSpecification().width;
    uint32_t height = _texture->getSpecification().height;
    float fx        = uv.x * (width - 1);
    float fy        = uv.y * (height - 1);

    int x0 = static_cast<int>(glm::floor(fx));
    int y0 = static_cast<int>(glm::floor(fy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float tx = fx - x0;
    float ty = fy - y0;

    float w00 = (1 - tx) * (1 - ty);
    float w10 = tx * (1 - ty);
    float w01 = (1 - tx) * ty;
    float w11 = tx * ty;

    _texture->writeTexel(w00 * val, glm::ivec2(x0, y0));
    _texture->writeTexel(w10 * val, glm::ivec2(x1, y0));
    _texture->writeTexel(w01 * val, glm::ivec2(x0, y1));
    _texture->writeTexel(w11 * val, glm::ivec2(x1, y1));
}

template<typename T, TexelWriteMode write_mode>
template<typename uv_t>
ATCG_HOST_DEVICE void InterpolationWriter<T, write_mode, TextureFilterMode::LINEAR>::_write_3d(const T& val,
                                                                                               const uv_t& uv)
{
    uint32_t width  = _texture->getSpecification().width;
    uint32_t height = _texture->getSpecification().height;
    uint32_t depth  = _texture->getSpecification().depth;
    float fx        = uv.x * (width - 1);
    float fy        = uv.y * (height - 1);
    float fz        = uv.z * (depth - 1);

    int x0 = static_cast<int>(glm::floor(fx));
    int y0 = static_cast<int>(glm::floor(fy));
    int z0 = static_cast<int>(glm::floor(fz));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    float tx = fx - x0;
    float ty = fy - y0;
    float tz = fz - z0;

    float w000 = (1 - tx) * (1 - ty) * (1 - tz);
    float w100 = tx * (1 - ty) * (1 - tz);
    float w010 = (1 - tx) * ty * (1 - tz);
    float w110 = tx * ty * (1 - tz);
    float w001 = (1 - tx) * (1 - ty) * tz;
    float w101 = tx * (1 - ty) * tz;
    float w011 = (1 - tx) * ty * tz;
    float w111 = tx * ty * tz;

    _texture->writeTexel(w000 * val, glm::ivec3(x0, y0, z0));
    _texture->writeTexel(w100 * val, glm::ivec3(x1, y0, z0));
    _texture->writeTexel(w010 * val, glm::ivec3(x0, y1, z0));
    _texture->writeTexel(w110 * val, glm::ivec3(x1, y1, z0));
    _texture->writeTexel(w001 * val, glm::ivec3(x0, y0, z1));
    _texture->writeTexel(w101 * val, glm::ivec3(x1, y0, z1));
    _texture->writeTexel(w011 * val, glm::ivec3(x0, y1, z1));
    _texture->writeTexel(w111 * val, glm::ivec3(x1, y1, z1));
}

template<typename T>
ATCG_INLINE ATCG_HOST_DEVICE void TexelUpdater<TexelWriteMode::DEFAULT>::operator()(T* data, const T& val) const
{
    *data = val;
}

template<typename T>
ATCG_INLINE ATCG_DEVICE void TexelUpdater<TexelWriteMode::ATOMIC_ADD>::operator()(T* data, const T& val) const
{
    if constexpr(std::is_integral_v<T>)
    {
        atomicAdd((int*)data, (int)val);    // TODO?
    }
    else if constexpr(std::is_floating_point_v<T>)
    {
        atcg::globalAtomicAdd((float*)data, (float)val);
    }
    else
    {
        printf("Atomic add not supported for this type\n");
    }
}

}    // namespace atcg