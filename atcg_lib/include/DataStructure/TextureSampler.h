#pragma once

#include <Core/glm.h>
#include <Core/CUDA.h>
#include <Renderer/TextureSpecification.h>

namespace atcg
{

template<typename T>
struct is_supported_texel_type : std::false_type
{
};

template<>
struct is_supported_texel_type<float> : std::true_type
{
};
template<>
struct is_supported_texel_type<int32_t> : std::true_type
{
};
template<>
struct is_supported_texel_type<glm::vec2> : std::true_type
{
};
template<>
struct is_supported_texel_type<glm::vec3> : std::true_type
{
};
template<>
struct is_supported_texel_type<glm::vec4> : std::true_type
{
};

template<typename T>
constexpr bool is_supported_texel_type_v = is_supported_texel_type<T>::value;

template<typename T>
class TextureSampler
{
public:
    TextureSampler() = default;

    TextureSampler(void* data, const TextureSpecification& spec);

    ATCG_HOST_DEVICE T read(const glm::vec2& uv) const;

    ATCG_HOST_DEVICE T texel_fetch(const glm::ivec2& texel) const;

    ATCG_HOST_DEVICE void write(const T& val, const glm::ivec2& texel);

    ATCG_HOST_DEVICE void* getTexelPtr(const glm::ivec2& texel);

    ATCG_INLINE ATCG_HOST_DEVICE T operator()(const glm::vec2& uv) const { return read(uv); }

    ATCG_INLINE ATCG_HOST_DEVICE const TextureSpecification getSpecification() const { return _spec; }

private:
    ATCG_HOST_DEVICE glm::vec2 _clamp_uv(const glm::vec2& uv) const;

    ATCG_HOST_DEVICE T _read_interpolated(const glm::vec2& uv) const;

    ATCG_HOST_DEVICE T _read_nearest(const glm::vec2& uv) const;

    ATCG_HOST_DEVICE T _read_linear(const glm::vec2& uv) const;

private:
    void* _data                = nullptr;
    TextureSpecification _spec = {};

    static_assert(is_supported_texel_type_v<T>,
                  "TextureSampler only supports float, int32_t, glm::vec2, glm::vec3, glm::vec4");
};
}    // namespace atcg
#include "../../src/DataStructure/TextureSamplerDetail.h"