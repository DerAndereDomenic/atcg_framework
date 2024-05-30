#pragma once

#include <Core/glm.h>
#include <Core/Memory.h>
#include <Renderer/Buffer.h>
#include <Renderer/VertexArray.h>
#include <Renderer/Shader.h>
#include <Renderer/PerspectiveCamera.h>
#include <Renderer/ShaderManager.h>
#include <Renderer/Framebuffer.h>
#include <DataStructure/Graph.h>


namespace atcg
{
class Scene;
class Entity;

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
     * @brief Get the current clear color
     *
     * @return The clear color
     */
    static glm::vec4 getClearColor();

    /**
     * @brief Set the size of rendered points
     *
     * @param size The size
     */
    static void setPointSize(const float& size);

    /**
     * @brief Set the size of rendered lines
     *
     * @param size The size
     */
    static void setLineSize(const float& size);

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
     * @brief Set the viewport according to the current main screen render buffer
     *
     */
    static void setDefaultViewport();

    /**
     * @brief Set a skybox
     *
     * @param skybox An equirectangular representation of the skybox
     */
    static void setSkybox(const atcg::ref_ptr<Image>& skybox);

    /**
     * @brief If a skybox is set.
     *
     * @return True if there is a skybox set.
     */
    static bool hasSkybox();

    /**
     * @brief Remove the skybox
     */
    static void removeSkybox();

    /**
     * @brief Return the equirectangular skybox texture
     *
     * @return A pointer to the texture (only is valid if hasSkybox() == true)
     */
    static atcg::ref_ptr<Texture2D> getSkyboxTexture();

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
     * @brief Render a mesh
     *
     * The default draw mode is "base". It applys slight shading based on the vertex normals.
     * An optional color can be given to color the whole mesh with a constant color.
     * If given a custom shader, color is ignored except if the shader variable "flat_color" is used.
     *
     * @param mesh The mesh
     * @param camera The camera
     * @param model The optional model matrix
     * @param color An optional color
     * @param shader The shader
     * @param draw_mode The draw mode
     */
    static void draw(const atcg::ref_ptr<Graph>& mesh,
                     const atcg::ref_ptr<Camera>& camera = {},
                     const glm::mat4& model              = glm::mat4(1),
                     const glm::vec3& color              = glm::vec3(1),
                     const atcg::ref_ptr<Shader>& shader = atcg::ShaderManager::getShader("base"),
                     DrawMode draw_mode                  = DrawMode::ATCG_DRAW_MODE_TRIANGLE);

    /**
     * @brief Render an entity
     *
     * @param entity The entity to render
     * @param camera The camera
     */
    static void draw(Entity entity, const atcg::ref_ptr<Camera>& camera = {});

    /**
     * @brief Render a scene
     *
     * @param scene The scene to render
     * @param camera The camera
     */
    static void draw(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera = {});

    /**
     * @brief Draw camera frustrums
     *
     * @param scene The scene
     * @param camera The camera
     */
    static void drawCameras(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera = {});

    /**
     * @brief Draw Circle
     *
     * @param position The position
     * @param radius The radius
     * @param thickness The thickness of the circle
     * @param color The color
     * @param camera The camera
     */
    static void drawCircle(const glm::vec3& position,
                           const float& radius,
                           const float& thickness,
                           const glm::vec3& color,
                           const atcg::ref_ptr<Camera>& camera = {});

    /**
     * @brief Draws a CAD grid with three resolutions (0.1, 1, 10)
     *
     * @param camera The camera
     * @param transparency The transparency of the grid
     */
    static void drawCADGrid(const atcg::ref_ptr<Camera>& camera, const float& transparency = 0.6f);

    /**
     * @brief Get the framebuffer object that is used by the renderer
     *
     * @return The fbo
     */
    static atcg::ref_ptr<Framebuffer> getFramebuffer();

    /**
     * @brief Get the entity index that was rendered onto the given pixel
     *
     * @param mouse The mouse position
     * @return The entity id
     */
    static int getEntityIndex(const glm::vec2& mouse);

    /**
     * @brief Take a screenshot and save it to disk
     *
     * @param scene The scene
     * @param camera The camera
     * @param width The output width. Height is calculated from the camera's aspect ratio
     * @param path The output path
     */
    static void screenshot(const atcg::ref_ptr<Scene>& scene,
                           const atcg::ref_ptr<Camera>& camera,
                           const uint32_t width,
                           const std::string& path);

    /**
     * @brief Take a screenshot and return it as tensor
     *
     * @param scene The scene
     * @param camera The camera
     * @param width The output width. Height is calculated from the camera's aspect ratio
     *
     * @return The pixel data as tensor
     */
    static torch::Tensor
    screenshot(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera, const uint32_t width);

    /**
     * @brief Get a buffer representing the color attachement of the screen frame buffer.
     * @note This copies memory between GPU and CPU
     *
     * @return The buffer containing the frame image.
     */
    static std::vector<uint8_t> getFrame();

    /**
     * @brief Get a buffer representing the color attachement of given frame buffer
     * @note This copies memory between GPU and CPU
     *
     * @param fbo The framebuffer holding the image information
     *
     * @return The buffer containing the frame image.
     */
    static std::vector<uint8_t> getFrame(const atcg::ref_ptr<Framebuffer>& fbo);

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
     * @brief Toggle depth testing
     *
     * @param enable If it should be enabled or disabled
     */
    static void toggleDepthTesting(bool enable = true);

    /**
     * @brief Toggle face culling
     *
     * @param enable If it should be enabled or disabled
     */
    static void toggleCulling(bool enable = true);

    /**
     * @brief Set the cull face
     *
     * @param mode The culling mode
     */
    static void setCullFace(CullMode mode);

    /**
     * @brief Get the current frame counter
     *
     * @return The index of the current frame
     */
    static uint32_t getFrameCounter();

    /**
     * @brief Destroys the renderer instance
     */
    ATCG_INLINE static void destroy() { delete s_renderer; }


    ~Renderer();

private:
    Renderer();

    class Impl;
    atcg::scope_ptr<Impl> impl;

    static Renderer* s_renderer;
};
}    // namespace atcg