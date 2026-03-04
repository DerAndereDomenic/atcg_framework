#pragma once

#include <CuDiff/ext/glm/Wrap.h>

namespace atcg
{

template<typename T>
template<typename iuv_t>
ATCG_HOST_DEVICE void* TextureInterface<T>::getTexelPtr(const iuv_t& texel) const
{
    size_t index = toIndex(texel);
    if(_spec.isFloat() || _spec.isInt())
    {
        float* pixels = reinterpret_cast<float*>(_data);
        return (void*)&pixels[index];
    }
    else
    {
        uint8_t* pixels = reinterpret_cast<uint8_t*>(_data);
        return (void*)&pixels[index];
    }
}

template<typename T>
ATCG_INLINE ATCG_HOST_DEVICE size_t TextureInterface<T>::toIndex(const glm::ivec2& texel) const
{
    return size_t((texel.y * _spec.width + texel.x) * _spec.numChannels());
}

template<typename T>
ATCG_INLINE ATCG_HOST_DEVICE size_t TextureInterface<T>::toIndex(const glm::ivec3& texel) const
{
    return size_t((texel.x + texel.y * _spec.width + texel.z * _spec.width * _spec.height) * _spec.numChannels());
}

template<typename T>
TextureSampler<T>::TextureSampler(void* data, const TextureSpecification& spec) : TextureInterface<T>(data, spec)
{
}

template<typename T>
template<typename uv_t>
ATCG_INLINE ATCG_HOST_DEVICE auto TextureSampler<T>::read(const uv_t& uv) const
{
    auto _uv = clamp_uv(uv);
    auto x   = _read_interpolated(_uv);
    return x;
}

template<typename T>
template<typename iuv_t>
ATCG_INLINE ATCG_HOST_DEVICE T TextureSampler<T>::texel_fetch(const iuv_t& texel) const
{
    size_t index = toIndex(texel);
    if(_spec.isFloat() || _spec.isInt())
    {
        const float* pixels = reinterpret_cast<const float*>(_data);
        if constexpr((std::is_same_v<T, float>) || (std::is_same_v<T, int32_t>))
        {
            return (T)pixels[index];
        }
        else
        {
            T result(0);
            for(uint32_t i = 0; i < vec_traits<T>::dim; ++i)
            {
                result[i] = pixels[index + i];
            }
            return result;
        }
    }
    else
    {
        const uint8_t* pixels = reinterpret_cast<const uint8_t*>(_data);
        if constexpr(std::is_same_v<T, float>)
        {
            return (float)pixels[index] / 255.0f;
        }
        else
        {
            T result(0);
            for(uint32_t i = 0; i < vec_traits<T>::dim; ++i)
            {
                result[i] = (float)pixels[index + i] / 255.0f;
            }
            return result;
        }
    }
}

template<typename T>
template<typename uv_t>
ATCG_INLINE ATCG_HOST_DEVICE uv_t TextureSampler<T>::clamp_uv(const uv_t& uv) const
{
    if constexpr(CuDiff::is_dual_v<uv_t>)
    {
        return CuDiff::clamp(uv, CuDiff::dual_value_type_t<uv_t>(0), CuDiff::dual_value_type_t<uv_t>(1));
    }
    else
    {
        switch(_spec.sampler.wrap_mode)
        {
            case TextureWrapMode::BORDER:
            {
            };
            // break; Not implemented yet
            case TextureWrapMode::CLAMP_TO_EDGE:
            {
                if constexpr(std::is_same_v<uv_t, glm::vec2>)
                {
                    return glm::vec2(glm::clamp(uv.x, 0.0f, 1.0f), glm::clamp(uv.y, 0.0f, 1.0f));
                }
                else if constexpr(std::is_same_v<uv_t, glm::vec3>)
                {
                    return glm::vec3(glm::clamp(uv.x, 0.0f, 1.0f),
                                     glm::clamp(uv.y, 0.0f, 1.0f),
                                     glm::clamp(uv.z, 0.0f, 1.0f));
                }
            };
            break;
            case TextureWrapMode::REPEAT:
            {
                return glm::abs(glm::fract(uv));
            };
            break;
        }

        return uv;
    }
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
            return _read_linear(uv);
        }
        break;
        case TextureFilterMode::NEAREST:
        {
            return _read_nearest(uv);
        }
        break;
    }

    return decltype(_read_linear(uv))(T(0));
}

template<typename T>
template<typename uv_t>
ATCG_HOST_DEVICE auto TextureSampler<T>::_read_nearest(const uv_t& uv) const
{
    if constexpr(std::is_same_v<CuDiff::dual_value_type_t<uv_t>, glm::vec2>)
    {
        return _read_nearest_2d(uv);
    }
    else if constexpr(std::is_same_v<CuDiff::dual_value_type_t<uv_t>, glm::vec3>)
    {
        return _read_nearest_3d(uv);
    }
    else
    {
        static_assert(!(std::is_same_v<uv_t, uv_t>), "Unsupported UV type");
        return T(0);
    }
}

template<typename T>
template<typename uv_t>
ATCG_HOST_DEVICE auto TextureSampler<T>::_read_linear(const uv_t& uv) const
{
    if constexpr(std::is_same_v<CuDiff::dual_value_type_t<uv_t>, glm::vec2>)
    {
        return _read_linear_2d(uv);
    }
    else if constexpr(std::is_same_v<CuDiff::dual_value_type_t<uv_t>, glm::vec3>)
    {
        return _read_linear_3d(uv);
    }
    else
    {
        static_assert(!(std::is_same_v<uv_t, uv_t>), "Unsupported UV type");
        return T(0);
    }
}

template<typename T>
template<typename uv_t>
ATCG_HOST_DEVICE auto TextureSampler<T>::_read_nearest_2d(const uv_t& uv) const
{
    auto [uv_x, uv_y] = CuDiff::unwrap(uv);
    uint32_t texel_x  = (uint32_t)(CuDiff::value_of(uv_x) * _spec.width);
    uint32_t texel_y  = (uint32_t)(CuDiff::value_of(uv_y) * _spec.height);

    texel_x = CuDiff::clamp(texel_x, uint32_t(0), uint32_t(_spec.width - 1));
    texel_y = CuDiff::clamp(texel_y, uint32_t(0), uint32_t(_spec.height - 1));

    auto texel = texel_fetch(glm::ivec2(texel_x, texel_y));
    if constexpr(CuDiff::is_dual_v<uv_t>)
    {
        using R  = CuDiff::Dual<CuDiff::dual_component_count<uv_t>::num_variables, T>;
        R result = R(texel);
        // Nearest neighbor sampling is not differentiable, so we set the derivatives to 0
        return result;
    }
    else
    {
        return texel;
    }
}

template<typename T>
template<typename uv_t>
ATCG_HOST_DEVICE auto TextureSampler<T>::_read_nearest_3d(const uv_t& uv) const
{
    auto [uv_x, uv_y, uv_z] = CuDiff::unwrap(uv);
    uint32_t texel_x        = (uint32_t)(CuDiff::value_of(uv_x) * _spec.width);
    uint32_t texel_y        = (uint32_t)(CuDiff::value_of(uv_y) * _spec.height);
    uint32_t texel_z        = (uint32_t)(CuDiff::value_of(uv_z) * _spec.depth);

    texel_x = glm::clamp(texel_x, 0u, _spec.width - 1u);
    texel_y = glm::clamp(texel_y, 0u, _spec.height - 1u);
    texel_z = glm::clamp(texel_z, 0u, _spec.depth - 1u);

    auto texel = texel_fetch(glm::ivec2(texel_x, texel_y));
    if constexpr(CuDiff::is_dual_v<uv_t>)
    {
        using R  = CuDiff::Dual<CuDiff::dual_component_count<uv_t>::num_variables, T>;
        R result = R(texel);
        // Nearest neighbor sampling is not differentiable, so we set the derivatives to 0
        return result;
    }
    else
    {
        return texel;
    }
}

template<typename T>
template<typename uv_t>
ATCG_HOST_DEVICE auto TextureSampler<T>::_read_linear_2d(const uv_t& uv) const
{
    auto [ux, uy] = CuDiff::unwrap(uv);

    auto fx = ux * (_spec.width - 1);
    auto fy = uy * (_spec.height - 1);

    int x0 = static_cast<int>(glm::floor(CuDiff::value_of(fx)));
    int y0 = static_cast<int>(glm::floor(CuDiff::value_of(fy)));
    int x1 = glm::min(x0 + 1, (int)_spec.width - 1);    // TODO: wrap
    int y1 = glm::min(y0 + 1, (int)_spec.height - 1);

    auto tx = fx - (float)x0;
    auto ty = fy - (float)y0;

    T c00 = texel_fetch(glm::ivec2(x0, y0));
    T c10 = texel_fetch(glm::ivec2(x1, y0));
    T c01 = texel_fetch(glm::ivec2(x0, y1));
    T c11 = texel_fetch(glm::ivec2(x1, y1));

    auto cx0 = (1.0f - tx) * c00 + c10 * tx;
    auto cx1 = (1.0f - tx) * c01 + c11 * tx;
    auto res = (1.0f - ty) * cx0 + cx1 * ty;

    return res;
}

template<typename T>
template<typename uv_t>
ATCG_HOST_DEVICE auto TextureSampler<T>::_read_linear_3d(const uv_t& uv) const
{
    auto [ux, uy, uz] = CuDiff::unwrap(uv);

    auto fx = ux * (_spec.width - 1);
    auto fy = uy * (_spec.height - 1);
    auto fz = uz * (_spec.depth - 1);

    int x0 = static_cast<int>(glm::floor(fx));
    int y0 = static_cast<int>(glm::floor(fy));
    int z0 = static_cast<int>(glm::floor(fz));
    int x1 = glm::min(x0 + 1, (int)_spec.width - 1);    // TODO: wrap
    int y1 = glm::min(y0 + 1, (int)_spec.height - 1);
    int z1 = glm::min(z0 + 1, (int)_spec.depth - 1);

    auto tx = fx - x0;
    auto ty = fy - y0;
    auto tz = fz - z0;

    T c000 = texel_fetch(glm::ivec3(x0, y0, z0));
    T c100 = texel_fetch(glm::ivec3(x1, y0, z0));
    T c010 = texel_fetch(glm::ivec3(x0, y1, z0));
    T c110 = texel_fetch(glm::ivec3(x1, y1, z0));
    T c001 = texel_fetch(glm::ivec3(x0, y0, z1));
    T c101 = texel_fetch(glm::ivec3(x1, y0, z1));
    T c011 = texel_fetch(glm::ivec3(x0, y1, z1));
    T c111 = texel_fetch(glm::ivec3(x1, y1, z1));

    auto cx00 = c000 * (1.0f - tx) + c100 * tx;
    auto cx10 = c010 * (1.0f - tx) + c110 * tx;
    auto cx01 = c001 * (1.0f - tx) + c101 * tx;
    auto cx11 = c011 * (1.0f - tx) + c111 * tx;

    auto cxy0 = cx00 * (1.0f - ty) + cx10 * ty;
    auto cxy1 = cx01 * (1.0f - ty) + cx11 * ty;

    auto res = cxy0 * (1.0f - tz) + cxy1 * tz;

    return res;
}

template<typename T>
TextureWriter<T>::TextureWriter(void* data, const TextureSpecification& spec) : TextureInterface<T>(data, spec)
{
}

template<typename T>
ATCG_INLINE ATCG_HOST_DEVICE void TextureWriter<T>::write(const T& val, const glm::ivec2& texel)
{
    size_t index = (texel.y * _spec.width + texel.x) * _spec.numChannels();

    if(_spec.isFloat())
    {
        float* pixels = reinterpret_cast<float*>(_data);
        if constexpr(std::is_same_v<T, float>)
        {
            pixels[index] = val;
        }
        else
        {
            for(uint32_t i = 0; i < vec_traits<T>::dim; i++)
                pixels[index + i] = val[i];
        }
    }
    else if(_spec.isInt())
    {
        if constexpr(std::is_same_v<T, int32_t>)
        {
            int32_t* pixels = reinterpret_cast<int32_t*>(_data);
            pixels[index]   = val;
        }
        else
        {
            static_assert(!(std::is_same_v<T, int32_t>), "Can only write integer as int32_t");
        }
    }
    else
    {
        uint8_t* pixels = reinterpret_cast<uint8_t*>(_data);
        if constexpr(std::is_same_v<T, float>)
        {
            pixels[index] = static_cast<uint8_t>(glm::clamp(val, 0.0f, 1.0f) * 255.0f);
        }
        else
        {
            for(uint32_t i = 0; i < vec_traits<T>::dim; i++)
            {
                pixels[index + i] = static_cast<uint8_t>(glm::clamp(val[i], 0.0f, 1.0f) * 255.0f);
            }
        }
    }
}

template<typename T>
ATCG_INLINE ATCG_DEVICE void TextureWriter<T>::writeAtomicAdd(const T& val, const glm::ivec2& texel)
{
    size_t index = (texel.y * _spec.width + texel.x) * _spec.numChannels();

    if(_spec.isFloat())
    {
        float* pixels = reinterpret_cast<float*>(_data);
        if constexpr(std::is_same_v<T, float>)
        {
            atomicAdd(pixels + index, val);
        }
        else
        {
            for(uint32_t i = 0; i < vec_traits<T>::dim; i++)
                atomicAdd(pixels + index + i, val[i]);
        }
    }
    else if(_spec.isInt())
    {
        if constexpr(std::is_same_v<T, int32_t>)
        {
            int32_t* pixels = reinterpret_cast<int32_t*>(_data);
            atomicAdd(pixels + index, val);
        }
        else
        {
            static_assert(!(std::is_same_v<T, int32_t>), "Can only write integer as int32_t");
        }
    }
    else
    {
        printf("Atomic add not supported for normalized uint8_t textures\n");
    }
}

}    // namespace atcg