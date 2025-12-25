#pragma once

namespace atcg
{
/**
 * @brief An enum defining draw modes.
 *
 */
enum DrawMode
{
    ATCG_DRAW_MODE_TRIANGLE,          // Draw as standard mesh
    ATCG_DRAW_MODE_POINTS,            // Draw as points (screen space)
    ATCG_DRAW_MODE_POINTS_SPHERE,     // Draw points as spheres
    ATCG_DRAW_MODE_EDGES,             // Draw edges
    ATCG_DRAW_MODE_EDGES_CYLINDER,    // Draw edges as 3D cylinders
    ATCG_DRAW_MODE_INSTANCED          // Draw a standard mesh instanced
};

/**
 * @brief An enum defining cull modes.
 *
 */
enum CullMode
{
    ATCG_FRONT_FACE_CULLING,
    ATCG_BACK_FACE_CULLING,
    ATCG_BOTH_FACE_CULLING
};

struct RenderState
{
    CullMode cull_mode         = ATCG_BACK_FACE_CULLING;
    bool culling_enabled       = false;
    bool depth_testing_enabled = true;
    glm::vec4 clear_color      = glm::vec4(0.1f, 0.1f, 0.1f, 1.0f);
    float point_size           = 1.0f;
    float line_size            = 1.0f;
    glm::ivec4 viewport;

    RenderState() = default;

    ATCG_INLINE RenderState setCullMode(CullMode mode)
    {
        cull_mode = mode;
        return *this;
    }

    ATCG_INLINE RenderState enableCulling(bool enable = true)
    {
        culling_enabled = enable;
        return *this;
    }

    ATCG_INLINE RenderState enableDepthTesting(bool enable = true)
    {
        depth_testing_enabled = enable;
        return *this;
    }

    ATCG_INLINE RenderState setClearColor(const glm::vec4& color)
    {
        clear_color = color;
        return *this;
    }

    ATCG_INLINE RenderState setPointSize(const float& size)
    {
        point_size = size;
        return *this;
    }

    ATCG_INLINE RenderState setLineSize(const float& size)
    {
        line_size = size;
        return *this;
    }

    ATCG_INLINE RenderState setViewport(const glm::ivec4& vp)
    {
        viewport = vp;
        return *this;
    }
};
}    // namespace atcg