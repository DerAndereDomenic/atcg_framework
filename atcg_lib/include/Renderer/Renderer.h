#pragma once

#include <Renderer/Buffer.h>
#include <Renderer/VertexArray.h>
#include <Renderer/Shader.h>
#include <Renderer/PerspectiveCamera.h>
#include <DataStructure/Mesh.h>
#include <DataStructure/Grid.h>
#include <DataStructure/PointCloud.h>
#include <Renderer/ShaderManager.h>

#include <memory>

#include <glm/glm.hpp>

namespace atcg
{
/**
 * @brief An enum defining draw modes.
 *
 */
enum DrawMode
{
    ATCG_DRAW_MODE_TRIANGLE,         // Draw as standard mesh
    ATCG_DRAW_MODE_POINTS,           // Draw as points (screen space)
    ATCG_DRAW_MODE_POINTS_SPHERE,    // Draw points as spheres
    ATCG_DRAW_MODE_EDGES,            // Draw edges
};

/**
 * @brief This class models a renderer
 */
class Renderer
{
public:
    /**
     * @brief Initializes the renderer (should not be called by the client!)
     *
     * @param width The width
     * @param height The height
     */
    static void init(uint32_t width, uint32_t height);

    /**
     * @brief Finished the currently drawn frame (should not be called by client!)
     */
    static void finishFrame();

    /**
     * @brief Set the clear color
     *
     * @param color The clear color
     */
    static void setClearColor(const glm::vec4& color);

    /**
     * @brief Set the size of rendered points
     *
     * @param size The size
     */
    static void setPointSize(const float& size);

    /**
     * @brief Change the viewport of the renderer
     *
     * @param x The viewport x location
     * @param y The viewport y location
     * @param width The width
     * @param height The height
     */
    static void setViewport(const uint32_t& x, const uint32_t& y, const uint32_t& width, const uint32_t& height);

    /**
     * @brief Change the size of the renderer
     *
     * @param width The width
     * @param height The height
     */
    static void resize(const uint32_t& width, const uint32_t& height);

    /**
     * @brief Use the default screen fbo
     */
    static void useScreenBuffer();

    /**
     * @brief Render a vao
     * NEEDS to have an index buffer
     *
     * The default draw mode is "base". It applys slight shading based on the vertex normals.
     * An optional color can be given to color the whole mesh with a constant color.
     * If given a custom shader, color is ignored except if the shader variable "flat_color" is used.
     *
     * @param vao The vertex array
     * @param camera The camera
     * @param color An optional color
     * @param shader The shader
     * @param draw_mode The draw mode
     */
    static void draw(const std::shared_ptr<VertexArray>& vao,
                     const std::shared_ptr<Camera>& camera = {},
                     const glm::vec3& color                = glm::vec3(1),
                     const std::shared_ptr<Shader>& shader = atcg::ShaderManager::getShader("base"),
                     DrawMode draw_mode                    = DrawMode::ATCG_DRAW_MODE_TRIANGLE);

    /**
     * @brief Render a mesh
     *
     * The default draw mode is "base". It applys slight shading based on the vertex normals.
     * An optional color can be given to color the whole mesh with a constant color.
     * If given a custom shader, color is ignored except if the shader variable "flat_color" is used.
     *
     * @param mesh The mesh
     * @param camera The camera
     * @param color An optional color
     * @param shader The shader
     * @param draw_mode The draw mode
     */
    static void draw(const std::shared_ptr<Mesh>& mesh,
                     const std::shared_ptr<Camera>& camera = {},
                     const glm::vec3& color                = glm::vec3(1),
                     const std::shared_ptr<Shader>& shader = atcg::ShaderManager::getShader("base"),
                     DrawMode draw_mode                    = DrawMode::ATCG_DRAW_MODE_TRIANGLE);

    /**
     * @brief Draw a pointcloud
     *
     * The default draw mode is "base". It applys slight shading based on the vertex normals.
     * An optional color can be given to color the whole mesh with a constant color.
     * If given a custom shader, color is ignored except if the shader variable "flat_color" is used.
     *
     * @param cloud The pointcloud
     * @param camera The camera
     * @param color An optional color
     * @param shader The shader
     * @param draw_mode The draw mode
     */
    static void draw(const std::shared_ptr<PointCloud>& cloud,
                     const std::shared_ptr<Camera>& camera = {},
                     const glm::vec3& color                = glm::vec3(1),
                     const std::shared_ptr<Shader>& shader = atcg::ShaderManager::getShader("base"),
                     DrawMode draw_mode                    = DrawMode::ATCG_DRAW_MODE_POINTS);

    /**
     * @brief Draw Circle
     *
     * @param position The position
     * @param radius The radius
     * @param color The color
     * @param camera The camera
     */
    static void drawCircle(const glm::vec3& position,
                           const float& radius,
                           const glm::vec3& color,
                           const std::shared_ptr<Camera>& camera = {});

    static std::vector<uint8_t> getFrame();
    /**
     * @brief Generate a Z-Buffer buffer.
     *
     * @return The depth buffer
     */
    static std::vector<float> getZBuffer();

    /**
     * @brief Clear the currently bound framebuffer
     */
    static void clear();

    /**
     * @brief Destroys the renderer instance
     */
    inline static void destroy() { delete s_renderer; }

    ~Renderer();

private:
    Renderer();

    class Impl;
    std::unique_ptr<Impl> impl;

    static Renderer* s_renderer;
};
}    // namespace atcg