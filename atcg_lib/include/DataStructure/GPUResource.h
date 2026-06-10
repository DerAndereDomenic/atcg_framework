#pragma once

#include <Core/API.h>
#include <Renderer/TextureSpecification.h>
#include <Renderer/Texture.h>
#include <Renderer/Buffer.h>
#include <DataStructure/TorchUtils.h>
#include <Renderer/CompileData.h>

#include <vector>
#include <variant>

namespace atcg
{
// A resource that can be used in a render pass. This can be a texture, a buffer or a tensor.
using PhysicalResource = std::variant<atcg::ref_ptr<VertexBuffer>, atcg::ref_ptr<Texture>, torch::Tensor>;

enum class ResourceType
{
    Texture,
    Buffer,
    Tensor,
};

enum class TextureSizeHint
{
    NONE,
    FULL_FRAMEBUFFER,
    HALF_FRAMEBUFFER,
    QUARTER_FRAMEBUFFER,
    DYNAMIC
};

/**
 * @brief A helper class to model the size of a texture. This can either be a fixed dimension or a hint that gets
 * converted to a dimension at runtime.
 */
struct TextureSize
{
    /**
     * @brief Construct a new Texture Size object
     */
    TextureSize() = default;

    /**
     * @brief Construct a new Texture Size object from a hint.
     * Using this constructor makes this object not store an explicit dimension, but a hint that gets converted to a
     * dimension at runtime.
     *
     * @param hint The hint
     */
    TextureSize(TextureSizeHint hint) : hint(hint) {}

    /**
     * @brief Construct a new Texture Size object from an explicit dimension.
     * Using this constructor makes this object store an explicit dimension.
     *
     * @param dimension The dimension
     */
    TextureSize(uint32_t dimension) : hint(TextureSizeHint::NONE), dimension(dimension) {}

    /**
     * @brief Get the dimension of the texture size.
     *
     * @return The dimension
     */
    operator uint32_t() const { return dimension; }

    /**
     * @brief Get if this texture size is a hint or an explicit dimension.
     *
     * @return True if this texture holds an explicit dimension, false if it holds a hint
     */
    operator bool() const { return hint == TextureSizeHint::NONE; }

    /**
     * @brief Set the dimension of the texture size. This makes this object hold an explicit dimension.
     *
     * @param dim The dimension
     */
    void operator=(uint32_t dim)
    {
        hint      = TextureSizeHint::NONE;
        dimension = dim;
    }

    /**
     * @brief Set the hint of the texture size. This makes this object hold a hint.
     *
     * @param new_hint The hint
     */
    void operator=(TextureSizeHint new_hint)
    {
        hint      = new_hint;
        dimension = 0;
    }

    /**
     * @brief Convert the hint to a dimension using the provided framebuffer dimension. If this object already holds an
     * explicit dimension, this function just returns it.
     *
     * @param dim The framebuffer dimension to use for conversion if this object holds a hint
     * @return The dimension
     */
    uint32_t convert(const uint32_t dim) const
    {
        switch(hint)
        {
            case TextureSizeHint::NONE:
                return dimension;
            case TextureSizeHint::FULL_FRAMEBUFFER:
                return dim;
            case TextureSizeHint::HALF_FRAMEBUFFER:
                return dim / 2;
            case TextureSizeHint::QUARTER_FRAMEBUFFER:
                return dim / 4;
            case TextureSizeHint::DYNAMIC:
                return dim;
        }
        return 0;
    }

private:
    TextureSizeHint hint = TextureSizeHint::NONE;
    uint32_t dimension   = 0;
};

struct TextureResourceDescription
{
    TextureType type                    = TextureType::TEXTURE_2D;
    TextureFormat format                = TextureFormat::RGBA;
    TextureSamplerSpecification sampler = {};
    TextureSize width                   = TextureSizeHint::FULL_FRAMEBUFFER;
    TextureSize height                  = TextureSizeHint::FULL_FRAMEBUFFER;
    TextureSize depth                   = 0;
};

struct BufferResourceDescription
{
    size_t size = 0;
    BufferLayout layout;
};

struct TensorResourceDescription
{
    torch::TensorOptions options;
    std::vector<int> size;
};

struct ResourceDescription
{
    ResourceType type;

    TextureResourceDescription texture;
    BufferResourceDescription buffer;
    TensorResourceDescription tensor;
};

/**
 * @brief Create a physical resource from a resource description. The resource is created on the GPU and can be used in
 * render passes.
 *
 * @param ctx The compile data
 * @param desc The resource description
 *
 * @return The created physical resource
 */
ATCG_API PhysicalResource createResource(const CompileData& ctx, const ResourceDescription& desc);
}    // namespace atcg