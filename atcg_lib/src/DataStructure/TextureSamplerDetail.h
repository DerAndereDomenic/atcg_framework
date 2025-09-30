#pragma once

namespace atcg
{

template<typename T>
TextureSampler<T>::TextureSampler(void* data, const TextureSpecification& spec) : _data(data),
                                                                                  _spec(spec)
{
}

template<typename T>
ATCG_INLINE ATCG_HOST_DEVICE T TextureSampler<T>::read(const glm::vec2& uv) const
{
    glm::vec2 _uv = _clamp_uv(uv);
    T x           = _read_interpolated(_uv);
    return x;
}

template<typename T>
ATCG_INLINE ATCG_HOST_DEVICE T TextureSampler<T>::texel_fetch(const glm::ivec2& texel) const
{
    size_t index = (texel.y * _spec.width + texel.x) * _spec.numChannels();
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
ATCG_INLINE ATCG_HOST_DEVICE void TextureSampler<T>::write(const T& val, const glm::ivec2& texel)
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
ATCG_HOST_DEVICE void* TextureSampler<T>::getTexelPtr(const glm::ivec2& texel)
{
    size_t index = (texel.y * _spec.width + texel.x) * _spec.numChannels();
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
ATCG_INLINE ATCG_HOST_DEVICE glm::vec2 TextureSampler<T>::_clamp_uv(const glm::vec2& uv) const
{
    switch(_spec.sampler.wrap_mode)
    {
        case TextureWrapMode::BORDER:
        {
        };
        // break; Not implemented yet
        case TextureWrapMode::CLAMP_TO_EDGE:
        {
            return glm::vec2(glm::clamp(uv.x, 0.0f, 1.0f), glm::clamp(uv.y, 0.0f, 1.0f));
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

template<typename T>
ATCG_HOST_DEVICE T TextureSampler<T>::_read_interpolated(const glm::vec2& uv) const
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

    return T(0);
}

template<typename T>
ATCG_HOST_DEVICE T TextureSampler<T>::_read_nearest(const glm::vec2& uv) const
{
    uint32_t texel_x = (uint32_t)(uv.x * _spec.width);
    uint32_t texel_y = (uint32_t)(uv.y * _spec.height);

    texel_x = glm::clamp(texel_x, 0u, _spec.width - 1u);
    texel_y = glm::clamp(texel_y, 0u, _spec.height - 1u);

    return texel_fetch(glm::ivec2(texel_x, texel_y));
}

template<typename T>
ATCG_HOST_DEVICE T TextureSampler<T>::_read_linear(const glm::vec2& uv) const
{
    float fx = uv.x * (_spec.width - 1);
    float fy = uv.y * (_spec.height - 1);

    int x0 = static_cast<int>(glm::floor(fx));
    int y0 = static_cast<int>(glm::floor(fy));
    int x1 = glm::min(x0 + 1, (int)_spec.width - 1);    // TODO: wrap
    int y1 = glm::min(y0 + 1, (int)_spec.height - 1);

    float tx = fx - x0;
    float ty = fy - y0;

    T c00 = texel_fetch(glm::ivec2(x0, y0));
    T c10 = texel_fetch(glm::ivec2(x1, y0));
    T c01 = texel_fetch(glm::ivec2(x0, y1));
    T c11 = texel_fetch(glm::ivec2(x1, y1));

    T cx0 = glm::mix(c00, c10, tx);
    T cx1 = glm::mix(c01, c11, tx);
    T res = glm::mix(cx0, cx1, ty);

    return res;
}

}    // namespace atcg