#pragma once

#include <Core/glm.h>
#include <Core/Memory.h>
#include <Renderer/Context.h>
#include <Renderer/Buffer.h>
#include <Renderer/VertexArray.h>
#include <Renderer/Shader.h>
#include <Renderer/PerspectiveCamera.h>
#include <Renderer/ShaderManager.h>
#include <Renderer/Framebuffer.h>
#include <DataStructure/Graph.h>
#include <Scene/Entity.h>
#include <Renderer/Material.h>

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
 * @brief This class models a renderer. This should be used if you are trying to create multiple renderer instances
 * (e.g. for multi threading). Per default, use atcg::Renderer to use the default renderer.
 */
class RendererSystem
{
public:
    /**
     * @brief Initializes the renderer (should not be called by the client!)
     *
     * @param width The width
     * @param height The height
     * @param context The opengl context
     * @param shader_manager The shader manager associated with this renderer
     */
    void init(uint32_t width,
              uint32_t height,
              const atcg::ref_ptr<Context>& context,
              const atcg::ref_ptr<ShaderManagerSystem>& shader_manager);

    /**
     * @brief Use this instance of the renderer.
     * This only needs to be called if multiple renderer instances are in use or multi thread rendering is desired. This
     * function will make the underlying context current for the thread.
     */
    void use();

    /**
     * @brief Finished the currently drawn frame (should not be called by client!)
     */
    void finishFrame();

    /**
     * @brief Stalls the CPU until all rendering requests are finished
     */
    void finish() const;

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
     * @brief Set the number of MSAA samples
     *
     * @param num_samples The number of MSAA samples
     */
    void setMSAA(uint32_t num_samples);

    /**
     * @brief Get the current number of MSAA samples
     *
     * @return The number of samples
     */
    uint32_t getMSAA() const;

    /**
     * @brief Toggle MSAA (on per default).
     * This can be used to "disable" the internal msaa framebuffer. This can be useful if it is desired to directly draw
     * into the final framebuffer of the renderer that can be accessed with getFramebuffer(). If MSAA is enabled, this
     * framebuffer will be overwritten with the result of the msaa buffer by blitting. This will not be done if MSAA is
     * toggled off.
     *
     * @param enable If MSAA should be enabled
     */
    void toggleMSAA(const bool enable = true);

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
     * @brief Get the current viewport dimensions
     *
     * @return A vec4 containing the viewport (x,y,width,height)
     */
    glm::vec4 getViewport() const;

    /**
     * @brief Preprocess a skybox
     *
     * @param skybox An equirectangular representation of the skybox
     * @param skybox_cubemap The output skybox as a cubemap
     * @param irradiance_cubemap The output irradiance cubemap
     * @param prefiltered_cubemap The output prefiltered cubemap for ibl
     */
    void processSkybox(const atcg::ref_ptr<Texture2D>& skybox,
                       const atcg::ref_ptr<TextureCube>& skybox_cubemap,
                       const atcg::ref_ptr<TextureCube>& irradiance_cubemap,
                       const atcg::ref_ptr<TextureCube>& prefiltered_cubemap);

    /**
     * @brief Change the size of the renderer
     *
     * @param width The width
     * @param height The height
     */
    void resize(const uint32_t& width, const uint32_t& height);

    /**
     * @brief Use the default screen fbo.
     * Per default this will be a MSAA framebuffer. If MSAA is disabled using toggleMSAA, this function will bind the
     * normal framebuffer that is used in the end to render to the screen.
     */
    void useScreenBuffer() const;

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

    /**
     * @brief Clear the currently bound framebuffer
     */
    void clear() const;

    /**
     * @brief Render a mesh
     *
     * The default draw mode is "base". It applys slight shading based on the vertex normals.
     * An optional color can be given to color the whole mesh with a constant color.
     * If given a custom shader, color is ignored except if the shader variable "flat_color" is used.
     *
     * For ATCG_DRAW_MDOE_SPHERE The custom shader has to implement an instance rendering scheme like the "base" shader.
     * If no custom shader is to be used, stick to the "base" shader. For ATCG_DRAW_MODE_EDGE and
     * ATCG_DRAW_MODE_CYLINDER, you can pass nullptr for the shader to use the built in "edge" and "edge_cylinder"
     * shader. If a custom shader should be used, it has to implement the basic instancing the is used in the
     * aforementioned shaders.
     *
     * @param mesh The mesh
     * @param camera The camera
     * @param model The optional model matrix
     * @param color An optional color
     * @param shader The shader
     * @param draw_mode The draw mode
     * @param material The material
     * @param entity_id The entity id
     */
    void draw(const atcg::ref_ptr<Graph>& mesh,
              const atcg::ref_ptr<Camera>& camera                    = {},
              const glm::mat4& model                                 = glm::mat4(1),
              const glm::vec3& color                                 = glm::vec3(1),
              const atcg::ref_ptr<Shader>& shader                    = atcg::ShaderManager::getShader("base"),
              DrawMode draw_mode                                     = DrawMode::ATCG_DRAW_MODE_TRIANGLE,
              const std::optional<atcg::ref_ptr<Material>>& material = {},
              const uint32_t entity_id                               = -1);

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
     * @brief Draw a skybox
     *
     * @param skybox The skybox cubemap
     * @param camera The camera
     */
    void drawSkybox(const atcg::ref_ptr<TextureCube>& skybox_cubemap, const atcg::ref_ptr<Camera>& camera);

    /**
     * @brief Draw camera frustrums
     *
     * @param scene The scene
     * @param camera The camera
     */
    void drawCameras(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera = {});

    /**
     * @brief Draw light sources
     *
     * @param scene The scene
     * @param camera The camera
     */
    void drawLights(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera = {});

    /**
     * @brief Draws a CAD grid with three resolutions (0.1, 1, 10)
     *
     * @param camera The camera
     * @param transparency The transparency of the grid
     */
    void drawCADGrid(const atcg::ref_ptr<Camera>& camera, const float& transparency = 0.6f);

    /**
     * @brief Get the framebuffer objects that is used by the renderer.
     * Depending on if MSAA is enabled or not, this may be a MSAA framebuffer that can only be used for additional
     * rendering tasks. Data reads can not be performed in this case.
     *
     * @return The framebuffer of last frame
     */
    atcg::ref_ptr<Framebuffer> getFramebuffer() const;

    /**
     * @brief Get the framebuffer objects that is used by the renderer.
     * This framebuffer does not represent the current frame but the last frame after finishFrame() was called. To get
     * the current frame data, refer to getFramebufferMSAA().
     * Any direct draw call done to this framebuffer will be overwritten at the end of the current frame by the blit
     * operation on the msaa buffer. If direct rendering to this framebuffer is desired, MSAA has to be toggle off using
     * toggleMSAA.
     *
     * @return The framebuffer of last frame
     */
    atcg::ref_ptr<Framebuffer> getResolvedFramebuffer() const;

    /**
     * @brief Get the framebuffer object that is used by the renderer.
     * This is a multi sample framebuffer that can only be used for drawing operations. If you want to read rendered
     * data, either use getFramebufefr() which contains the render of the last frame. If you need the current
     * state of the framebuffer for this frame, you have to manually blit it.
     *
     * @return The fbo
     */
    atcg::ref_ptr<Framebuffer> getFramebufferMSAA() const;

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
     * @brief Get the shader manager associated with this renderer
     *
     * @return The shader manager
     */
    atcg::ref_ptr<ShaderManagerSystem> getShaderManager() const;

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
 * @param context The context
 * @param shader_manager The shader manager associated with this renderer
 */
ATCG_INLINE void init(uint32_t width,
                      uint32_t height,
                      const atcg::ref_ptr<Context>& context,
                      const atcg::ref_ptr<ShaderManagerSystem>& shader_manager)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->init(width, height, context, shader_manager);
}

/**
 * @brief Use this instance of the renderer.
 * This only needs to be called if multiple renderer instances are in use or multi thread rendering is desired. This
 * function will make the underlying context current for the thread.
 */
ATCG_INLINE void use()
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->use();
}

/**
 * @brief Finished the currently drawn frame (should not be called by client!)
 */
ATCG_INLINE void finishFrame()
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->finishFrame();
}

/**
 * @brief Stalls the CPU until all rendering requests are finished
 */
ATCG_INLINE void finish()
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->finish();
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
 * @brief Set the number of MSAA samples
 *
 * @param num_samples The number of MSAA samples
 */
ATCG_INLINE void setMSAA(uint32_t num_samples)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->setMSAA(num_samples);
}

/**
 * @brief Get the current number of MSAA samples
 *
 * @return The number of samples
 */
ATCG_INLINE uint32_t getMSAA()
{
    return SystemRegistry::instance()->getSystem<RendererSystem>()->getMSAA();
}

/**
 * @brief Toggle MSAA (on per default).
 * This can be used to "disable" the internal msaa framebuffer. This can be useful if it is desired to directly draw
 * into the final framebuffer of the renderer that can be accessed with getFramebuffer(). If MSAA is enabled, this
 * framebuffer will be overwritten with the result of the msaa buffer by blitting. This will not be done if MSAA is
 * toggled off.
 *
 * @param enable If MSAA should be enabled
 */
ATCG_INLINE void toggleMSAA(const bool enable = true)
{
    return SystemRegistry::instance()->getSystem<RendererSystem>()->toggleMSAA(enable);
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
 * @brief Get the current viewport dimensions
 *
 * @return A vec4 containing the viewport (x,y,width,height)
 */
ATCG_INLINE glm::vec4 getViewport()
{
    return SystemRegistry::instance()->getSystem<RendererSystem>()->getViewport();
}

/**
 * @brief Preprocess a skybox
 *
 * @param skybox An equirectangular representation of the skybox
 * @param skybox_cubemap The output skybox as a cubemap
 * @param irradiance_cubemap The output irradiance cubemap
 * @param prefiltered_cubemap The output prefiltered cubemap for ibl
 */
ATCG_INLINE void processSkybox(const atcg::ref_ptr<Texture2D>& skybox,
                               const atcg::ref_ptr<TextureCube>& skybox_cubemap,
                               const atcg::ref_ptr<TextureCube>& irradiance_cubemap,
                               const atcg::ref_ptr<TextureCube>& prefiltered_cubemap)
{
    return SystemRegistry::instance()->getSystem<RendererSystem>()->processSkybox(skybox,
                                                                                  skybox_cubemap,
                                                                                  irradiance_cubemap,
                                                                                  prefiltered_cubemap);
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
 * @brief Use the default screen fbo.
 * Per default this will be a MSAA framebuffer. If MSAA is disabled using toggleMSAA, this function will bind the
 * normal framebuffer that is used in the end to render to the screen.
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
 * For ATCG_DRAW_MDOE_SPHERE The custom shader has to implement an instance rendering scheme like the "base" shader.
 * If no custom shader is to be used, stick to the "base" shader. For ATCG_DRAW_MODE_EDGE and
 * ATCG_DRAW_MODE_CYLINDER, you can pass nullptr for the shader to use the built in "edge" and "edge_cylinder"
 * shader. If a custom shader should be used, it has to implement the basic instancing the is used in the
 * aforementioned shaders.
 *
 * @param mesh The mesh
 * @param camera The camera
 * @param model The optional model matrix
 * @param color An optional color
 * @param shader The shader
 * @param draw_mode The draw mode
 * @param material The material
 * @param entity_id The entity id
 */
ATCG_INLINE void draw(const atcg::ref_ptr<Graph>& mesh,
                      const atcg::ref_ptr<Camera>& camera     = {},
                      const glm::mat4& model                  = glm::mat4(1),
                      const glm::vec3& color                  = glm::vec3(1),
                      const atcg::ref_ptr<Shader>& shader     = atcg::ShaderManager::getShader("base"),
                      DrawMode draw_mode                      = DrawMode::ATCG_DRAW_MODE_TRIANGLE,
                      const std::optional<atcg::ref_ptr<Material>>& material = {},
                      const uint32_t entity_id                = -1)
{
    SystemRegistry::instance()
        ->getSystem<RendererSystem>()
        ->draw(mesh, camera, model, color, shader, draw_mode, material, entity_id);
}

/**
 * @brief Draw a skybox
 *
 * @param skybox The skybox cubemap
 * @param camera The camera
 */
ATCG_INLINE void drawSkybox(const atcg::ref_ptr<TextureCube>& skybox_cubemap, const atcg::ref_ptr<Camera>& camera)
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->drawSkybox(skybox_cubemap, camera);
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
 * @brief Draw light sources
 *
 * @param scene The scene
 * @param camera The camera
 */
ATCG_INLINE void drawLights(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera = {})
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->drawLights(scene, camera);
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
 * @brief Get the framebuffer objects that is used by the renderer.
 * This framebuffer does not represent the current frame but the last frame after finishFrame() was called. To get
 * the current frame data, refer to getFramebufferMSAA().
 * Any direct draw call done to this framebuffer will be overwritten at the end of the current frame by the blit
 * operation on the msaa buffer. If direct rendering to this framebuffer is desired, MSAA has to be toggle off using
 * toggleMSAA.
 *
 * @return The framebuffer of last frame
 */
ATCG_INLINE atcg::ref_ptr<Framebuffer> getFramebuffer()
{
    return SystemRegistry::instance()->getSystem<RendererSystem>()->getFramebuffer();
}

/**
 * @brief Get the framebuffer objects that is used by the renderer.
 * This framebuffer does not represent the current frame but the last frame after finishFrame() was called. To get
 * the current frame data, refer to getFramebufferMSAA().
 * Any direct draw call done to this framebuffer will be overwritten at the end of the current frame by the blit
 * operation on the msaa buffer. If direct rendering to this framebuffer is desired, MSAA has to be toggle off using
 * toggleMSAA.
 *
 * @return The framebuffer of last frame
 */
ATCG_INLINE atcg::ref_ptr<Framebuffer> getResolvedFramebuffer()
{
    return SystemRegistry::instance()->getSystem<RendererSystem>()->getResolvedFramebuffer();
}

/**
 * @brief Get the framebuffer object that is used by the renderer.
 * This is a multi sample framebuffer that can only be used for drawing operations. If you want to read rendered
 * data, either use getFramebuffer() which contains the render of the last frame. If you need the current state of
 * the framebuffer for this frame, you have to manually blit it.
 *
 * @return The fbo
 */
ATCG_INLINE atcg::ref_ptr<Framebuffer> getFramebufferMSAA()
{
    return SystemRegistry::instance()->getSystem<RendererSystem>()->getFramebufferMSAA();
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

/**
 * @brief Get the shader manager associated with this renderer
 *
 * @return The shader manager
 */
ATCG_INLINE atcg::ref_ptr<ShaderManagerSystem> getShaderManager()
{
    SystemRegistry::instance()->getSystem<RendererSystem>()->getShaderManager();
}
}    // namespace Renderer

}    // namespace atcg