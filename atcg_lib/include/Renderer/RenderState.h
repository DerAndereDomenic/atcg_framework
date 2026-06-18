#pragma once

#include <Core/API.h>
namespace atcg
{

/**
 * @brief An enum defining cull modes.
 *
 */
enum CullMode
{
    ATCG_FRONT_FACE_CULLING = 0,
    ATCG_BACK_FACE_CULLING  = 1,
    ATCG_BOTH_FACE_CULLING  = 2,
    ATCG_NO_CULLING         = 3
};

enum class PrimitiveTopology
{
    ATCG_POINTS,
    ATCG_TRIANGLES,
    ATCG_LINES
};

enum class DepthFunction
{
    ATCG_LESS,
    ATCG_LEQUAL,
    ATCG_EQUAL,
    ATCG_GREATER,
    ATCG_GEQUAL,
    ATCG_ALWAYS,
    ATCG_NEVER,
    ATCG_NOTEQUAL
};

struct ATCG_API DepthState
{
    bool depth_testing_enabled   = true;
    bool depth_write_enabled     = true;
    DepthFunction depth_function = DepthFunction::ATCG_LESS;

    DepthState() = default;

    ATCG_INLINE DepthState setDepthFunction(DepthFunction func)
    {
        depth_function = func;
        return *this;
    }

    ATCG_INLINE DepthState enableDepthTesting(bool enable = true)
    {
        depth_testing_enabled = enable;
        return *this;
    }

    ATCG_INLINE DepthState enableDepthWrite(bool enable = true)
    {
        depth_write_enabled = enable;
        return *this;
    }

    ATCG_INLINE bool operator==(const DepthState& other) const
    {
        return depth_testing_enabled == other.depth_testing_enabled &&
               depth_write_enabled == other.depth_write_enabled && depth_function == other.depth_function;
    }
};

struct ATCG_API BlendState
{
    bool blend_enabled = true;

    BlendState() = default;

    ATCG_INLINE BlendState enableBlending(bool enable = true)
    {
        blend_enabled = enable;
        return *this;
    }

    ATCG_INLINE bool operator==(const BlendState& other) const { return blend_enabled == other.blend_enabled; }
};

struct ATCG_API RasterizerState
{
    CullMode cull_mode   = ATCG_BACK_FACE_CULLING;
    bool culling_enabled = false;
    DepthState depth_state;
    BlendState blend_state;
    float point_size = 1.0f;
    float line_size  = 1.0f;

    RasterizerState() = default;

    ATCG_INLINE RasterizerState setCullMode(CullMode mode)
    {
        cull_mode = mode;
        return *this;
    }

    ATCG_INLINE RasterizerState enableCulling(bool enable = true)
    {
        culling_enabled = enable;
        return *this;
    }

    ATCG_INLINE RasterizerState setDepthState(const DepthState& state)
    {
        depth_state = state;
        return *this;
    }

    ATCG_INLINE RasterizerState setBlendState(const BlendState& state)
    {
        blend_state = state;
        return *this;
    }

    ATCG_INLINE RasterizerState setPointSize(const float& size)
    {
        point_size = size;
        return *this;
    }

    ATCG_INLINE RasterizerState setLineSize(const float& size)
    {
        line_size = size;
        return *this;
    }

    ATCG_INLINE bool operator==(const RasterizerState& other) const
    {
        return cull_mode == other.cull_mode && culling_enabled == other.culling_enabled &&
               depth_state == other.depth_state && blend_state == other.blend_state && point_size == other.point_size &&
               line_size == other.line_size;
    }
};
}    // namespace atcg