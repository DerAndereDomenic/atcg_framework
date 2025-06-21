#pragma once

#include <Core/Platform.h>
#include <Renderer/Texture.h>

namespace atcg
{

/**
 * @brief A class to build textures
 */
class TextureBuilder
{
public:
    /**
     * @brief Default constructor
     */
    TextureBuilder() = default;

    /**
     * @brief Set the format
     *
     * @param format The texture format
     *
     * @return *this
     */
    ATCG_INLINE TextureBuilder setFormat(atcg::TextureFormat format)
    {
        _spec.format = format;
        return *this;
    }

    /**
     * @brief Set the sampler
     *
     * @param sampler The sampler
     *
     * @return *this
     */
    ATCG_INLINE TextureBuilder setSampler(atcg::TextureSampler sampler)
    {
        _spec.sampler = sampler;
        return *this;
    }

    /**
     * @brief Set the width
     *
     * @param width The texture width
     *
     * @return *this
     */
    ATCG_INLINE TextureBuilder setWidth(uint32_t width)
    {
        _spec.width = width;
        return *this;
    }

    /**
     * @brief Set the height
     *
     * @param height The texture height
     *
     * @return *this
     */
    ATCG_INLINE TextureBuilder setHeight(uint32_t height)
    {
        _spec.height = height;
        return *this;
    }

    /**
     * @brief Set the depth
     *
     * @param depth The texture depth
     *
     * @return *this
     */
    ATCG_INLINE TextureBuilder setDepth(uint32_t depth)
    {
        _spec.depth = depth;
        return *this;
    }

    /**
     * @brief Set the wrap mode
     *
     * @param wrap_mode The texture sampler's wrap mode
     *
     * @return *this
     */
    ATCG_INLINE TextureBuilder setWrapMode(atcg::TextureWrapMode wrap_mode)
    {
        _spec.sampler.wrap_mode = wrap_mode;
        return *this;
    }

    /**
     * @brief Set the filter mode
     *
     * @param format The texture sampler's filter mode
     *
     * @return *this
     */
    ATCG_INLINE TextureBuilder setFilterMode(atcg::TextureFilterMode filter_mode)
    {
        _spec.sampler.filter_mode = filter_mode;
        return *this;
    }

    /**
     * @brief Enable mip mapping
     *
     * @param mip_map If mip mapping should be enabled
     *
     * @return *this
     */
    ATCG_INLINE TextureBuilder setMipMap(bool mip_map)
    {
        _spec.sampler.mip_map = mip_map;
        return *this;
    }

    /**
     * @brief Get the texture specification
     *
     * @return The specification
     */
    ATCG_INLINE TextureSpecification getSpecification() const { return _spec; }

    /**
     * @brief Create the texture
     *
     * @tparam TextureT The texture type
     *
     * @return The texture containing the current information of the texture builder
     */
    template<typename TextureT>
    ATCG_INLINE atcg::ref_ptr<TextureT> create()
    {
        return TextureT::create(_spec);
    }

private:
    atcg::TextureSpecification _spec;
};
}    // namespace atcg