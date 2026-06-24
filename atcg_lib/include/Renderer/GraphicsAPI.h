#pragma once

#include <Core/API.h>
#include <Core/SystemRegistry.h>
#include <Renderer/Buffer.h>
#include <Renderer/Framebuffer.h>
#include <Renderer/GraphicsPipeline.h>
#include <Renderer/Texture.h>
#include <Renderer/VertexArray.h>

namespace atcg
{
/**
 * @brief This class models the rendering API
 */
class ATCG_API GraphicsAPI
{
public:
    /**
     * @brief Initializes the GraphicsAPI
     */
    void init();

    /**
     * @brief Begins a render pass
     *
     * @param target The target framebuffer
     */
    void beginRenderPass(const atcg::ref_ptr<Framebuffer>& target);

    /**
     * @brief Ends the current render pass
     */
    void endRenderPass();

    /**
     * @brief Bind a graphics pipeline.
     * This sets the state of the rasterizer, i.e., depth testing, blending, face culling, primitive topology, etc.
     *
     * @param pipeline The pipeline to bind
     */
    void bindPipeline(const GraphicsPipeline& pipeline);

    /**
     * @brief Set the viewport
     *
     * @param x The x position
     * @param y The y position
     * @param width The width
     * @param height The height
     */
    void setViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height);

    /**
     * @brief Get the current viewport
     *
     * @return The viewport as ivec4(x, y, width, height)
     */
    glm::ivec4 getViewport() const;

    /**
     * @brief Bind a vertex array object
     *
     * @param vao The vertex array to bind
     */
    void bindVertexArray(const atcg::ref_ptr<VertexArray>& vao);

    /**
     * @brief Bind a texture to the given slot
     *
     * @param slot The slot
     * @param texture The texture
     */
    void bindTexture(uint32_t slot, const atcg::ref_ptr<Texture>& texture);

    /**
     * @brief Bind a storage buffer to the given slot
     *
     * @param slot The slot
     * @param buffer The buffer
     */
    void bindStorageBuffer(uint32_t slot, const atcg::ref_ptr<VertexBuffer>& buffer);

    /**
     * @brief Draw call for simple vertex arrays
     *
     * @param vertexCount The number of vertices to draw
     */
    void draw(uint32_t vertexCount) const;

    /**
     * @brief Draw call for indexed vertex arrays
     *
     * @param indexCount The number of indices to draw
     */
    void drawIndexed(uint32_t indexCount) const;

    /**
     * @brief Draw call for instanced vertex arrays
     *
     * @param vertexCount The number of vertices per instance
     * @param nInstances The number of instances to draw
     */
    void drawInstanced(uint32_t vertexCount, uint32_t nInstances) const;

    /**
     * @brief Draw call for indexed and instanced vertex arrays
     *
     * @param indexCount The number of indices per instance
     * @param nInstances The number of instances to draw
     */
    void drawIndexedInstanced(uint32_t indexCount, uint32_t nInstances) const;

    /**
     * @brief Set the clear color
     *
     * @param color The color
     */
    void setClearColor(const glm::vec4& color);

    /**
     * @brief Set the clear depth
     *
     * @param depth The depth value
     */
    void setClearDepth(float depth = 1.0f);

    /**
     * @brief Clear the current framebuffer with the clear color
     */
    void clear();

    /**
     * @brief Forces the GPU to finish all operations
     */
    void finish() const;

    /**
     * @brief Get the total number of texture units available
     *
     * @return The number of texture units
     */
    int getTotalTextureUnits() const;

private:
    GraphicsPipeline _current_pipeline;
    atcg::ref_ptr<IndexBuffer> _current_ibo = nullptr;
    bool _started_render_pass               = false;

    struct TextureBinding
    {
        TextureBinding(uint32_t s, const atcg::ref_ptr<Texture>& t) : slot(s), texture(t) {}

        uint32_t slot;
        atcg::ref_ptr<Texture> texture;
    };

    std::vector<TextureBinding> _bound_textures = {};
};

namespace GraphicsCommand
{
/**
 * @brief Begins a render pass
 *
 * @param target The target framebuffer
 */
ATCG_INLINE void beginRenderPass(const atcg::ref_ptr<Framebuffer>& target)
{
    atcg::SystemRegistry::instance()->getSystem<GraphicsAPI>()->beginRenderPass(target);
}

/**
 * @brief Ends the current render pass
 */
ATCG_INLINE void endRenderPass()
{
    atcg::SystemRegistry::instance()->getSystem<GraphicsAPI>()->endRenderPass();
}

/**
 * @brief Bind a graphics pipeline.
 * This sets the state of the rasterizer, i.e., depth testing, blending, face culling, primitive topology, etc.
 *
 * @param pipeline The pipeline to bind
 */
ATCG_INLINE void bindPipeline(const GraphicsPipeline& pipeline)
{
    atcg::SystemRegistry::instance()->getSystem<GraphicsAPI>()->bindPipeline(pipeline);
}

/**
 * @brief Set the viewport
 *
 * @param x The x position
 * @param y The y position
 * @param width The width
 * @param height The height
 */
ATCG_INLINE void setViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
{
    atcg::SystemRegistry::instance()->getSystem<GraphicsAPI>()->setViewport(x, y, width, height);
}

/**
 * @brief Get the current viewport
 *
 * @return The viewport as ivec4(x, y, width, height)
 */
ATCG_INLINE glm::ivec4 getViewport()
{
    return atcg::SystemRegistry::instance()->getSystem<GraphicsAPI>()->getViewport();
}

/**
 * @brief Bind a vertex array object
 *
 * @param vao The vertex array to bind
 */
ATCG_INLINE void bindVertexArray(const atcg::ref_ptr<VertexArray>& vao)
{
    atcg::SystemRegistry::instance()->getSystem<GraphicsAPI>()->bindVertexArray(vao);
}

/**
 * @brief Bind a texture to the given slot
 *
 * @param slot The slot
 * @param texture The texture
 */
ATCG_INLINE void bindTexture(uint32_t slot, const atcg::ref_ptr<Texture>& texture)
{
    atcg::SystemRegistry::instance()->getSystem<GraphicsAPI>()->bindTexture(slot, texture);
}

/**
 * @brief Bind a storage buffer to the given slot
 *
 * @param slot The slot
 * @param buffer The buffer
 */
ATCG_INLINE void bindStorageBuffer(uint32_t slot, const atcg::ref_ptr<VertexBuffer>& buffer)
{
    atcg::SystemRegistry::instance()->getSystem<GraphicsAPI>()->bindStorageBuffer(slot, buffer);
}

/**
 * @brief Draw call for simple vertex arrays
 *
 * @param vertexCount The number of vertices to draw
 */
ATCG_INLINE void draw(uint32_t vertexCount)
{
    atcg::SystemRegistry::instance()->getSystem<GraphicsAPI>()->draw(vertexCount);
}

/**
 * @brief Draw call for indexed vertex arrays
 *
 * @param indexCount The number of indices to draw
 */
ATCG_INLINE void drawIndexed(uint32_t indexCount)
{
    atcg::SystemRegistry::instance()->getSystem<GraphicsAPI>()->drawIndexed(indexCount);
}

/**
 * @brief Draw call for instanced vertex arrays
 *
 * @param vertexCount The number of vertices per instance
 * @param nInstances The number of instances to draw
 */
ATCG_INLINE void drawInstanced(uint32_t vertexCount, uint32_t nInstances)
{
    atcg::SystemRegistry::instance()->getSystem<GraphicsAPI>()->drawInstanced(vertexCount, nInstances);
}

/**
 * @brief Draw call for indexed and instanced vertex arrays
 *
 * @param indexCount The number of indices per instance
 * @param nInstances The number of instances to draw
 */
ATCG_INLINE void drawIndexedInstanced(uint32_t indexCount, uint32_t nInstances)
{
    atcg::SystemRegistry::instance()->getSystem<GraphicsAPI>()->drawIndexedInstanced(indexCount, nInstances);
}

/**
 * @brief Set the clear color
 *
 * @param color The color
 */
ATCG_INLINE void setClearColor(const glm::vec4& color)
{
    atcg::SystemRegistry::instance()->getSystem<GraphicsAPI>()->setClearColor(color);
}

/**
 * @brief Set the clear depth
 *
 * @param depth The depth value
 */
ATCG_INLINE void setClearDepth(float depth = 1.0f)
{
    atcg::SystemRegistry::instance()->getSystem<GraphicsAPI>()->setClearDepth(depth);
}

/**
 * @brief Clear the current framebuffer with the clear color
 */
ATCG_INLINE void clear()
{
    atcg::SystemRegistry::instance()->getSystem<GraphicsAPI>()->clear();
}

/**
 * @brief Forces the GPU to finish all operations
 */
ATCG_INLINE void finish()
{
    atcg::SystemRegistry::instance()->getSystem<GraphicsAPI>()->finish();
}

/**
 * @brief Get the total number of texture units available
 *
 * @return The number of texture units
 */
ATCG_INLINE int getTotalTextureUnits()
{
    return atcg::SystemRegistry::instance()->getSystem<GraphicsAPI>()->getTotalTextureUnits();
}
}    // namespace GraphicsCommand

}    // namespace atcg