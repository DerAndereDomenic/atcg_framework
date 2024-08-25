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
#include <Scene/Entity.h>

namespace atcg
{
class Scene;

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
class RendererSystem
{
public:
    /**
     * @brief Initializes the renderer (should not be called by the client!)
     *
     * @param width The width
     * @param height The height
     */
    void init(uint32_t width, uint32_t height);

    /**
     * @brief Finished the currently drawn frame (should not be called by client!)
     */
    void finishFrame();

    /**
     * @brief Set the clear color
     *
     * @param color The clear color
     */
    void setClearColor(const glm::vec4& color);

    /**
     * @brief Get the current clear color
     *
     * @return The clear color
     */
    glm::vec4 getClearColor() const;

    /**
     * @brief Set the size of rendered points
     *
     * @param size The size
     */
    void setPointSize(const float& size);

    /**
     * @brief Set the size of rendered lines
     *
     * @param size The size
     */
    void setLineSize(const float& size);

    /**
     * @brief Change the viewport of the renderer
     *
     * @param x The viewport x location
     * @param y The viewport y location
     * @param width The width
     * @param height The height
     */
    void setViewport(const uint32_t& x, const uint32_t& y, const uint32_t& width, const uint32_t& height);

    /**
     * @brief Set the viewport according to the current main screen render buffer
     *
     */
    void setDefaultViewport();

    /**
     * @brief Set a skybox
     *
     * @param skybox An equirectangular representation of the skybox
     */
    void setSkybox(const atcg::ref_ptr<Image>& skybox);

    /**
     * @brief Set a skybox
     *
     * @param skybox An equirectangular representation of the skybox
     */
    void setSkybox(const atcg::ref_ptr<Texture2D>& skybox);

    /**
     * @brief If a skybox is set.
     *
     * @return True if there is a skybox set.
     */
    bool hasSkybox() const;

    /**
     * @brief Remove the skybox
     */
    void removeSkybox();

    /**
     * @brief Return the equirectangular skybox texture
     *
     * @return A pointer to the texture (only is valid if hasSkybox() == true)
     */
    atcg::ref_ptr<Texture2D> getSkyboxTexture() const;

    /**
     * @brief Get the cube map of the skybox
     *
     * @return The skybox cubemap
     */
    atcg::ref_ptr<TextureCube> getSkyboxCubemap() const;

    /**
     * @brief Change the size of the renderer
     *
     * @param width The width
     * @param height The height
     */
    void resize(const uint32_t& width, const uint32_t& height);

    /**
     * @brief Use the default screen fbo
     */
    void useScreenBuffer() const;

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
    void draw(const atcg::ref_ptr<Graph>& mesh,
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
    void draw(Entity entity, const atcg::ref_ptr<Camera>& camera = {});

    /**
     * @brief Render a scene
     *
     * @param scene The scene to render
     * @param camera The camera
     */
    void draw(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera = {});

    /**
     * @brief Draw camera frustrums
     *
     * @param scene The scene
     * @param camera The camera
     */
    void drawCameras(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera = {});

    /**
     * @brief Draw Circle
     *
     * @param position The position
     * @param radius The radius
     * @param thickness The thickness of the circle
     * @param color The color
     * @param camera The camera
     */
    void drawCircle(const glm::vec3& position,
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
    void drawCADGrid(const atcg::ref_ptr<Camera>& camera, const float& transparency = 0.6f);

    /**
     * @brief Display an image/texture in the main viewport. It is assumed that the image resolution is the same as the
     * viewport resolution. This function can be used to display results generated with a custom framebuffer. However,
     * entity selection is then not possible as the entity buffer is not filled. This function will assume that color
     * attachement 0 of the framebuffer is used.
     *
     * @param img The image to display
     */
    void drawImage(const atcg::ref_ptr<Framebuffer>& img);

    /**
     * @brief Display an image/texture in the main viewport. It is assumed that the image resolution is the same as the
     * viewport resolution. This function can be used to display results generated with a custom framebuffer. However,
     * entity selection is then not possible as the entity buffer is not filled.
     *
     * @param img The image to display
     */
    void drawImage(const atcg::ref_ptr<Texture2D>& img);

    /**
     * @brief Get the framebuffer object that is used by the renderer
     *
     * @return The fbo
     */
    atcg::ref_ptr<Framebuffer> getFramebuffer() const;

    /**
     * @brief Get the entity index that was rendered onto the given pixel
     *
     * @param mouse The mouse position
     * @return The entity id
     */
    int getEntityIndex(const glm::vec2& mouse) const;

    /**
     * @brief Take a screenshot and save it to disk
     *
     * @param scene The scene
     * @param camera The camera
     * @param width The output width. Height is calculated from the camera's aspect ratio
     * @param path The output path
     */
    void screenshot(const atcg::ref_ptr<Scene>& scene,
                    const atcg::ref_ptr<Camera>& camera,
                    const uint32_t width,
                    const std::string& path);

    /**
     * @brief Take a screenshot and save it to disk
     *
     * @param scene The scene
     * @param camera The camera
     * @param width The output width
     * @param height The output height
     * @param path The output path
     */
    void screenshot(const atcg::ref_ptr<Scene>& scene,
                    const atcg::ref_ptr<Camera>& camera,
                    const uint32_t width,
                    const uint32_t height,
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
    torch::Tensor
    screenshot(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera, const uint32_t width);

    /**
     * @brief Get a buffer representing the color attachement of the screen frame buffer.
     * @note This copies memory between GPU and CPU if device = CPU
     *
     * @param device The device
     *
     * @return The buffer containing the frame image.
     */
    torch::Tensor getFrame(const torch::DeviceType& device = atcg::GPU) const;

    /**
     * @brief Get the Z-buffer of the current frame as torch tensor
     * @note This function always does a GPU-CPU memcopy because depth maps can not be mapped from OpenGL to CUDA. If
     * device = GPU is specified, an additional memcpy from CPU to GPU is performed.
     *
     * @param device The device
     *
     * @return The depth buffer
     */
    torch::Tensor getZBuffer(const torch::DeviceType& device = atcg::GPU) const;

    /**
     * @brief Clear the currently bound framebuffer
     */
    void clear() const;

    /**
     * @brief Toggle depth testing
     *
     * @param enable If it should be enabled or disabled
     */
    void toggleDepthTesting(bool enable = true);

    /**
     * @brief Toggle face culling
     *
     * @param enable If it should be enabled or disabled
     */
    void toggleCulling(bool enable = true);

    /**
     * @brief Set the cull face
     *
     * @param mode The culling mode
     */
    void setCullFace(CullMode mode);

    /**
     * @brief Get the current frame counter
     *
     * @return The index of the current frame
     */
    uint32_t getFrameCounter() const;

    /**
     * @brief Generates a new texture ID that is not used yet.
     * The renderer internally uses a fixed set of render IDs that should not be used by the client. This function
     * guarantees that the given texture id is not yet taken (unless manual texture ids are created). If a texture id is
     * no longer in use, it should be released with pushTextureID.
     *
     * @return The texture ID
     */
    uint32_t popTextureID();

    /**
     * @brief Free a texture ID.
     * Should only be used on ids that were generated using popTextureID. Adding other IDs may break the system because
     * some ids are reserved for the internals of the Renderer.
     *
     * @param id The id generated by popTextureID
     */
    void pushTextureID(const uint32_t id);

    RendererSystem();

    ~RendererSystem();

private:
    class Impl;
    atcg::scope_ptr<Impl> impl;
};

/**
 * @brief This namespace encapsulates the default renderer of this framework.
 */
namespace Renderer
{
/**
 * @brief Initializes the renderer (should not be called by the client!)
 *
 * @param width The width
 * @param height The height
 */
ATCG_INLINE void init(uint32_t width, uint32_t height)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->init(width, height);
}

/**
 * @brief Finished the currently drawn frame (should not be called by client!)
 */
ATCG_INLINE void finishFrame()
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->finishFrame();
}

/**
 * @brief Set the clear color
 *
 * @param color The clear color
 */
ATCG_INLINE void setClearColor(const glm::vec4& color)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->setClearColor(color);
}

/**
 * @brief Get the current clear color
 *
 * @return The clear color
 */
ATCG_INLINE glm::vec4 getClearColor()
{
    return SystemRegistry::instance()->getSystem<RendererSystem>()->getClearColor();
}

/**
 * @brief Set the size of rendered points
 *
 * @param size The size
 */
ATCG_INLINE void setPointSize(const float& size)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->setPointSize(size);
}

/**
 * @brief Set the size of rendered lines
 *
 * @param size The size
 */
ATCG_INLINE void setLineSize(const float& size)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->setLineSize(size);
}

/**
 * @brief Change the viewport of the renderer
 *
 * @param x The viewport x location
 * @param y The viewport y location
 * @param width The width
 * @param height The height
 */
ATCG_INLINE void setViewport(const uint32_t& x, const uint32_t& y, const uint32_t& width, const uint32_t& height)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->setViewport(x, y, width, height);
}

/**
 * @brief Set the viewport according to the current main screen render buffer
 *
 */
ATCG_INLINE void setDefaultViewport()
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->setDefaultViewport();
}

/**
 * @brief Set a skybox
 *
 * @param skybox An equirectangular representation of the skybox
 */
ATCG_INLINE void setSkybox(const atcg::ref_ptr<Image>& skybox)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->setSkybox(skybox);
}

/**
 * @brief Set a skybox
 *
 * @param skybox An equirectangular representation of the skybox
 */
ATCG_INLINE void setSkybox(const atcg::ref_ptr<Texture2D>& skybox)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->setSkybox(skybox);
}

/**
 * @brief If a skybox is set.
 *
 * @return True if there is a skybox set.
 */
ATCG_INLINE bool hasSkybox()
{
    return SystemRegistry::instance()->getSystem<RendererSystem>()->hasSkybox();
}

/**
 * @brief Remove the skybox
 */
ATCG_INLINE void removeSkybox()
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->removeSkybox();
}

/**
 * @brief Return the equirectangular skybox texture
 *
 * @return A pointer to the texture (only is valid if hasSkybox() == true)
 */
ATCG_INLINE atcg::ref_ptr<Texture2D> getSkyboxTexture()
{
    return SystemRegistry::instance()->getSystem<RendererSystem>()->getSkyboxTexture();
}

/**
 * @brief Get the cube map of the skybox
 *
 * @return The skybox cubemap
 */
ATCG_INLINE atcg::ref_ptr<TextureCube> getSkyboxCubemap()
{
    return SystemRegistry::instance()->getSystem<RendererSystem>()->getSkyboxCubemap();
}

/**
 * @brief Change the size of the renderer
 *
 * @param width The width
 * @param height The height
 */
ATCG_INLINE void resize(const uint32_t& width, const uint32_t& height)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->resize(width, height);
}

/**
 * @brief Use the default screen fbo
 */
ATCG_INLINE void useScreenBuffer()
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->useScreenBuffer();
}

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
ATCG_INLINE void draw(const atcg::ref_ptr<Graph>& mesh,
                      const atcg::ref_ptr<Camera>& camera = {},
                      const glm::mat4& model              = glm::mat4(1),
                      const glm::vec3& color              = glm::vec3(1),
                      const atcg::ref_ptr<Shader>& shader = atcg::ShaderManager::getShader("base"),
                      DrawMode draw_mode                  = DrawMode::ATCG_DRAW_MODE_TRIANGLE)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->draw(mesh, camera, model, color, shader, draw_mode);
}

/**
 * @brief Render an entity
 *
 * @param entity The entity to render
 * @param camera The camera
 */
ATCG_INLINE void draw(Entity entity, const atcg::ref_ptr<Camera>& camera = {})
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->draw(entity, camera);
}

/**
 * @brief Render a scene
 *
 * @param scene The scene to render
 * @param camera The camera
 */
ATCG_INLINE void draw(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera = {})
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->draw(scene, camera);
}

/**
 * @brief Draw camera frustrums
 *
 * @param scene The scene
 * @param camera The camera
 */
ATCG_INLINE void drawCameras(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera = {})
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->drawCameras(scene, camera);
}

/**
 * @brief Draw Circle
 *
 * @param position The position
 * @param radius The radius
 * @param thickness The thickness of the circle
 * @param color The color
 * @param camera The camera
 */
ATCG_INLINE void drawCircle(const glm::vec3& position,
                            const float& radius,
                            const float& thickness,
                            const glm::vec3& color,
                            const atcg::ref_ptr<Camera>& camera = {})
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->drawCircle(position, radius, thickness, color, camera);
}

/**
 * @brief Draws a CAD grid with three resolutions (0.1, 1, 10)
 *
 * @param camera The camera
 * @param transparency The transparency of the grid
 */
ATCG_INLINE void drawCADGrid(const atcg::ref_ptr<Camera>& camera, const float& transparency = 0.6f)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->drawCADGrid(camera, transparency);
}

/**
 * @brief Display an image/texture in the main viewport. It is assumed that the image resolution is the same as the
 * viewport resolution. This function can be used to display results generated with a custom framebuffer. However,
 * entity selection is then not possible as the entity buffer is not filled. This function will assume that color
 * attachement 0 of the framebuffer is used.
 *
 * @param img The image to display
 */
ATCG_INLINE void drawImage(const atcg::ref_ptr<Framebuffer>& img)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->drawImage(img);
}

/**
 * @brief Display an image/texture in the main viewport. It is assumed that the image resolution is the same as the
 * viewport resolution. This function can be used to display results generated with a custom framebuffer. However,
 * entity selection is then not possible as the entity buffer is not filled.
 *
 * @param img The image to display
 */
ATCG_INLINE void drawImage(const atcg::ref_ptr<Texture2D>& img)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->drawImage(img);
}

/**
 * @brief Get the framebuffer object that is used by the renderer
 *
 * @return The fbo
 */
ATCG_INLINE atcg::ref_ptr<Framebuffer> getFramebuffer()
{
    return SystemRegistry::instance()->getSystem<RendererSystem>()->getFramebuffer();
}

/**
 * @brief Get the entity index that was rendered onto the given pixel
 *
 * @param mouse The mouse position
 * @return The entity id
 */
ATCG_INLINE int getEntityIndex(const glm::vec2& mouse)
{
    return SystemRegistry::instance()->getSystem<RendererSystem>()->getEntityIndex(mouse);
}

/**
 * @brief Take a screenshot and save it to disk
 *
 * @param scene The scene
 * @param camera The camera
 * @param width The output width. Height is calculated from the camera's aspect ratio
 * @param path The output path
 */
ATCG_INLINE void screenshot(const atcg::ref_ptr<Scene>& scene,
                            const atcg::ref_ptr<Camera>& camera,
                            const uint32_t width,
                            const std::string& path)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->screenshot(scene, camera, width, path);
}

/**
 * @brief Take a screenshot and save it to disk
 *
 * @param scene The scene
 * @param camera The camera
 * @param width The output width
 * @param height The output height
 * @param path The output path
 */
ATCG_INLINE void screenshot(const atcg::ref_ptr<Scene>& scene,
                            const atcg::ref_ptr<Camera>& camera,
                            const uint32_t width,
                            const uint32_t height,
                            const std::string& path)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->screenshot(scene, camera, width, height, path);
}

/**
 * @brief Take a screenshot and return it as tensor
 *
 * @param scene The scene
 * @param camera The camera
 * @param width The output width. Height is calculated from the camera's aspect ratio
 *
 * @return The pixel data as tensor
 */
ATCG_INLINE torch::Tensor
screenshot(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera, const uint32_t width)
{
    return SystemRegistry::instance()->getSystem<RendererSystem>()->screenshot(scene, camera, width);
}

/**
 * @brief Get a buffer representing the color attachement of the screen frame buffer.
 * @note This copies memory between GPU and CPU if device = CPU
 *
 * @param device The device
 *
 * @return The buffer containing the frame image.
 */
ATCG_INLINE torch::Tensor getFrame(const torch::DeviceType& device = atcg::GPU)
{
    return SystemRegistry::instance()->getSystem<RendererSystem>()->getFrame(device);
}

/**
 * @brief Get the Z-buffer of the current frame as torch tensor
 * @note This function always does a GPU-CPU memcopy because depth maps can not be mapped from OpenGL to CUDA. If
 * device = GPU is specified, an additional memcpy from CPU to GPU is performed.
 *
 * @param device The device
 *
 * @return The depth buffer
 */
ATCG_INLINE torch::Tensor getZBuffer(const torch::DeviceType& device = atcg::GPU)
{
    return SystemRegistry::instance()->getSystem<RendererSystem>()->getZBuffer(device);
}

/**
 * @brief Clear the currently bound framebuffer
 */
ATCG_INLINE void clear()
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->clear();
}

/**
 * @brief Toggle depth testing
 *
 * @param enable If it should be enabled or disabled
 */
ATCG_INLINE void toggleDepthTesting(bool enable = true)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->toggleDepthTesting(enable);
}

/**
 * @brief Toggle face culling
 *
 * @param enable If it should be enabled or disabled
 */
ATCG_INLINE void toggleCulling(bool enable = true)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->toggleCulling(enable);
}

/**
 * @brief Set the cull face
 *
 * @param mode The culling mode
 */
ATCG_INLINE void setCullFace(CullMode mode)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->setCullFace(mode);
}

/**
 * @brief Get the current frame counter
 *
 * @return The index of the current frame
 */
ATCG_INLINE uint32_t getFrameCounter()
{
    return SystemRegistry::instance()->getSystem<RendererSystem>()->getFrameCounter();
}

/**
 * @brief Generates a new texture ID that is not used yet.
 * The renderer internally uses a fixed set of render IDs that should not be used by the client. This function
 * guarantees that the given texture id is not yet taken (unless manual texture ids are created). If a texture id is
 * no longer in use, it should be released with pushTextureID.
 *
 * @return The texture ID
 */
ATCG_INLINE uint32_t popTextureID()
{
    return SystemRegistry::instance()->getSystem<RendererSystem>()->popTextureID();
}

/**
 * @brief Free a texture ID.
 * Should only be used on ids that were generated using popTextureID. Adding other IDs may break the system because
 * some ids are reserved for the internals of the Renderer.
 *
 * @param id The id generated by popTextureID
 */
ATCG_INLINE void pushTextureID(const uint32_t id)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->pushTextureID(id);
}
}    // namespace Renderer

}    // namespace atcg