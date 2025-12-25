#pragma once

#include <Renderer/Buffer.h>
#include <Renderer/Framebuffer.h>
#include <Renderer/GraphicsPipeline.h>
#include <Renderer/Texture.h>
#include <Renderer/VertexArray.h>

namespace atcg
{
class RenderAPI
{
public:
    void beginRenderPass(const atcg::ref_ptr<Framebuffer>& target);
    void endRenderPass();

    void setPipeline(const GraphicsPipeline& pipeline);

    void setViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height);

    void bindVertexArray(const atcg::ref_ptr<VertexArray>& vao);
    void bindTexture(uint32_t slot, const atcg::ref_ptr<Texture>& texture);

    void draw(uint32_t vertexCount);
    void drawIndexed(uint32_t indexCount);
    void drawInstanced(uint32_t vertexCount, uint32_t nInstances);
    void drawIndexedInstanced(uint32_t indexCount, uint32_t nInstances);


private:
    GraphicsPipeline _current_pipeline;
    atcg::ref_ptr<IndexBuffer> _current_ibo = nullptr;
};
}    // namespace atcg