#pragma once

#include <Core/API.h>
#include <Renderer/Shader.h>
#include <Renderer/RenderState.h>

namespace atcg
{
struct ATCG_API GraphicsPipeline
{
    atcg::ref_ptr<Shader> shader;
    RasterizerState rasterizer_state;
    PrimitiveTopology primitive_topology = PrimitiveTopology::ATCG_TRIANGLES;

    ATCG_INLINE GraphicsPipeline() = default;

    ATCG_INLINE GraphicsPipeline setShader(const atcg::ref_ptr<Shader>& shd)
    {
        shader = shd;
        return *this;
    }

    ATCG_INLINE GraphicsPipeline setRasterizerState(const RasterizerState& state)
    {
        rasterizer_state = state;
        return *this;
    }

    ATCG_INLINE GraphicsPipeline setPrimitiveTopology(const PrimitiveTopology& topology)
    {
        primitive_topology = topology;
        return *this;
    }

    ATCG_INLINE bool operator==(const GraphicsPipeline& other) const
    {
        return shader == other.shader && rasterizer_state == other.rasterizer_state &&
               primitive_topology == other.primitive_topology;
    }
};
}    // namespace atcg