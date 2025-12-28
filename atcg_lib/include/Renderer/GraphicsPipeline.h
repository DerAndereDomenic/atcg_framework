#pragma once

#include <Renderer/Shader.h>
#include <Renderer/RenderState.h>

namespace atcg
{
struct GraphicsPipeline
{
    atcg::ref_ptr<Shader> shader;
    RasterizerState render_state;
    PrimitiveTopology primitive_type = PrimitiveTopology::ATCG_TRIANGLES;

    ATCG_INLINE GraphicsPipeline() = default;

    ATCG_INLINE GraphicsPipeline setShader(const atcg::ref_ptr<Shader>& shd)
    {
        shader = shd;
        return *this;
    }

    ATCG_INLINE GraphicsPipeline setRasterizerState(const RasterizerState& state)
    {
        render_state = state;
        return *this;
    }

    ATCG_INLINE GraphicsPipeline setPrimitiveTopology(const PrimitiveTopology& topology)
    {
        primitive_type = topology;
        return *this;
    }

    ATCG_INLINE bool operator==(const GraphicsPipeline& other) const
    {
        return shader == other.shader && render_state == other.render_state && primitive_type == other.primitive_type;
    }
};
}    // namespace atcg