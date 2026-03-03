#pragma once

#include <Core/glm.h>
#include <Core/CUDA.h>
#include <Renderer/TextureSpecification.h>
#include <CuDiff/CuDiff.h>
#include <CuDiff/ext/glm/Function.h>
#include <CuDiff/ext/glm/Traits.h>

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
class TextureInterface
{
public:
    TextureInterface() = default;

    TextureInterface(void* data, const TextureSpecification& spec) : _data(data), _spec(spec) {}

    ATCG_INLINE ATCG_HOST_DEVICE const TextureSpecification getSpecification() const { return _spec; }

    ATCG_HOST_DEVICE void* getTexelPtr(const glm::ivec2& texel) const;

protected:
    void* _data                = nullptr;
    TextureSpecification _spec = {};
};

template<typename T>
class TextureSampler : public TextureInterface<T>
{
public:
    TextureSampler() = default;

    TextureSampler(void* data, const TextureSpecification& spec);

    template<typename uv_t>
    ATCG_HOST_DEVICE auto read(const uv_t& uv) const;

    ATCG_HOST_DEVICE T texel_fetch(const glm::ivec2& texel) const;

    ATCG_INLINE ATCG_HOST_DEVICE T operator()(const glm::vec2& uv) const { return read(uv); }

    template<typename uv_t>
    ATCG_HOST_DEVICE uv_t clamp_uv(const uv_t& uv) const;

private:
    template<typename uv_t>
    ATCG_HOST_DEVICE auto _read_interpolated(const uv_t& uv) const;

    template<typename uv_t>
    ATCG_HOST_DEVICE auto _read_nearest(const uv_t& uv) const;

    template<typename uv_t>
    ATCG_HOST_DEVICE auto _read_linear(const uv_t& uv) const;
};

template<typename T>
class TextureWriter
{
public:
    TextureWriter() = default;

    TextureWriter(void* data, const TextureSpecification& spec);

    ATCG_HOST_DEVICE void write(const T& val, const glm::ivec2& texel);

    ATCG_HOST_DEVICE void writeAtomicAdd(const T& val, const glm::ivec2& texel);

    ATCG_HOST_DEVICE void* getTexelPtr(const glm::ivec2& texel) const;

    ATCG_INLINE ATCG_HOST_DEVICE const TextureSpecification getSpecification() const { return _spec; }
};
}    // namespace atcg
#include "../../src/DataStructure/TextureSamplerDetail.h"