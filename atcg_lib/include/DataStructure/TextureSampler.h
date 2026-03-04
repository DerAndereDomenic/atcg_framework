#pragma once

#include <Core/glm.h>
#include <Core/CUDA.h>
#include <Renderer/TextureSpecification.h>

namespace atcg
{

enum class TexelWriteMode
{
    DEFAULT,
    ATOMIC_ADD
};

template<typename T>
class TextureInterface
{
public:
    TextureInterface() = default;

    TextureInterface(std::byte* data, const TextureSpecification& spec) : _data(data), _spec(spec) {}

    ATCG_INLINE ATCG_HOST_DEVICE const TextureSpecification getSpecification() const { return _spec; }

    template<typename iuv_t>
    ATCG_HOST_DEVICE std::byte* getTexelPtr(const iuv_t& texel) const;

    ATCG_HOST_DEVICE T convertData(const std::byte* data) const;

    template<typename iuv_t>
    ATCG_HOST_DEVICE T fetchTexel(const iuv_t& texel) const;

    template<typename iuv_t, TexelWriteMode write_mode = TexelWriteMode::DEFAULT>
    ATCG_HOST_DEVICE void writeTexel(const T& val, const iuv_t& texel);

protected:
    ATCG_HOST_DEVICE size_t toIndex(const glm::ivec2& texel) const;

    ATCG_HOST_DEVICE size_t toIndex(const glm::ivec3& texel) const;

    std::byte* _data           = nullptr;
    TextureSpecification _spec = {};
};

template<typename T>
class TextureSampler : public TextureInterface<T>
{
public:
    TextureSampler() = default;

    TextureSampler(std::byte* data, const TextureSpecification& spec) : TextureInterface<T>(data, spec) {}

    template<typename uv_t>
    ATCG_HOST_DEVICE auto read(const uv_t& uv) const;

private:
    template<typename uv_t>
    ATCG_HOST_DEVICE auto _read_interpolated(const uv_t& uv) const;
};

template<typename T, atcg::TextureFilterMode filter_mode>
struct InterpolationReader;

template<typename T>
struct InterpolationReader<T, TextureFilterMode::NEAREST>
{
    ATCG_HOST_DEVICE InterpolationReader(TextureInterface<T>* texture) : _texture(texture) {}

    template<typename uv_t>
    ATCG_HOST_DEVICE auto operator()(const uv_t& uv) const;

private:
    template<typename uv_t>
    ATCG_HOST_DEVICE auto _read_2d(const uv_t& uv) const;

    template<typename uv_t>
    ATCG_HOST_DEVICE auto _read_3d(const uv_t& uv) const;

    TextureInterface<T>* _texture;
};

template<typename T>
struct InterpolationReader<T, TextureFilterMode::LINEAR>
{
    ATCG_HOST_DEVICE InterpolationReader(TextureInterface<T>* texture) : _texture(texture) {}

    template<typename uv_t>
    ATCG_HOST_DEVICE auto operator()(const uv_t& uv) const;

private:
    template<typename uv_t>
    ATCG_HOST_DEVICE auto _read_2d(const uv_t& uv) const;

    template<typename uv_t>
    ATCG_HOST_DEVICE auto _read_3d(const uv_t& uv) const;

    TextureInterface<T>* _texture;
};

template<typename T>
class TextureWriter : public TextureInterface<T>
{
public:
    TextureWriter() = default;

    TextureWriter(std::byte* data, const TextureSpecification& spec) : TextureInterface<T>(data, spec) {}

    template<typename uv_t, TexelWriteMode write_mode = TexelWriteMode::DEFAULT>
    ATCG_HOST_DEVICE void write(const T& val, const uv_t& texel);
};

template<typename T, TexelWriteMode write_mode, atcg::TextureFilterMode filter_mode>
struct InterpolationWriter;

template<typename T, TexelWriteMode write_mode>
struct InterpolationWriter<T, write_mode, TextureFilterMode::NEAREST>
{
    InterpolationWriter(TextureInterface<T>* texture) : _texture(texture) {}

    template<typename uv_t>
    ATCG_HOST_DEVICE void operator()(const T& val, const uv_t& texel) const;

private:
    template<typename uv_t>
    ATCG_HOST_DEVICE void _write_2d(const T& val, const uv_t& texel);

    template<typename uv_t>
    ATCG_HOST_DEVICE void _write_3d(const T& val, const uv_t& texel);

    TextureInterface<T>* _texture;
};

template<typename T, TexelWriteMode write_mode>
struct InterpolationWriter<T, write_mode, TextureFilterMode::LINEAR>
{
    InterpolationWriter(TextureInterface<T>* texture) : _texture(texture) {}

    template<typename uv_t>
    ATCG_HOST_DEVICE void operator()(const T& val, const uv_t& texel) const;

private:
    template<typename uv_t>
    ATCG_HOST_DEVICE void _write_2d(const T& val, const uv_t& texel);

    template<typename uv_t>
    ATCG_HOST_DEVICE void _write_3d(const T& val, const uv_t& texel);

    TextureInterface<T>* _texture;
};

template<TexelWriteMode write_mode>
struct TexelUpdater;

template<>
struct TexelUpdater<TexelWriteMode::DEFAULT>
{
    template<typename T, typename iuv_t>
    ATCG_HOST_DEVICE void operator()(T* data, const T& val) const;
};

template<>
struct TexelUpdater<TexelWriteMode::ATOMIC_ADD>
{
    template<typename T, typename iuv_t>
    ATCG_HOST_DEVICE void operator()(T* data, const T& val) const;
};

}    // namespace atcg
#include "../../src/DataStructure/TextureSamplerDetail.h"