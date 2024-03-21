#pragma once

#include <Core/Memory.h>
#include <Core/glm.h>
#include <Renderer/Framebuffer.h>
#include <Renderer/PerspectiveCamera.h>
#include <Scene/Scene.h>

#include <functional>

namespace atcg
{
class VRRenderer
{
public:
    enum class Eye
    {
        LEFT  = 0,
        RIGHT = 1
    };

    enum class Role
    {
        HMD        = 0,
        LEFT_HAND  = 1,
        RIGHT_HAND = 2,
        INVALID    = -1
    };

    using EventCallbackFn = std::function<void(Event*)>;

    /**
     * @brief Initialize the VR renderer
     *
     * @param callback The callback function
     */
    static void init(const EventCallbackFn& callback);

    /**
     * @brief Initialize controller renderings.
     * This has to be called by the client. Adds controller entities to the scene
     *
     * These entities get updated automatically based on the current pose of the tracker every frame.
     *
     * @param scene The scene
     */
    static void initControllerMeshes(const atcg::ref_ptr<atcg::Scene>& scene);

    /**
     * @brief Gets called every frame and handles VR updates
     *
     * @param delta_time The delta frame time
     */
    static void onUpdate(const float delta_time);

    /**
     * @brief Do the tracking. This is a blocking call and should be called at the beginning of the rendering thread.
     * This is done by the application, not the client.
     */
    static void doTracking();

    /**
     * @brief This function pulls events from the headset and forwards them through the framework.
     *
     */
    static void emitEvents();

    /**
     * @brief Get the inverse view matrix of a specific eye
     *
     * @param eye The eye
     *
     * @return The view matrix
     */
    static glm::mat4 getInverseView(const Eye& eye);

    /**
     * @brief Get both inverse view matrices at the same time
     *
     * @return A tuple with the view matrices (Left, Right)
     */
    static std::tuple<glm::mat4, glm::mat4> getInverseViews();

    /**
     * @brief Get the projection matrix of a specific eye
     *
     * @param eye The eye
     *
     * @return The projection matrix
     */
    static glm::mat4 getProjection(const Eye& eye);

    /**
     * @brief Get borth projection matrices
     *
     * @return The projection matrices for both eyes (Left, Right)
     */
    static std::tuple<glm::mat4, glm::mat4> getProjections();

    /**
     * @brief Get the rendertarget framebuffers of a specific eye
     *
     * @param eye The eye
     *
     * @return The render target
     */
    static atcg::ref_ptr<Framebuffer> getRenderTarget(const Eye& eye);

    /**
     * @brief Get the render targets of both eyes
     *
     * @return A tuple with the framebuffers for each eye (Left, Right)
     */
    static std::tuple<atcg::ref_ptr<Framebuffer>, atcg::ref_ptr<Framebuffer>> getRenderTargets();

    /**
     * @brief Get the position of the Headset in tracking space
     *
     * @return The 3D position in tracking space
     */
    static glm::vec3 getPosition();

    /**
     * @brief Get the resolution of the Headset
     *
     * @return The width
     */
    static uint32_t width();

    /**
     * @brief Get the resolution of the Headset
     *
     * @return The height
     */
    static uint32_t height();

    /**
     * @brief Check if VR is available
     *
     * @return True if the HMD was found and the runtime could be initialized
     */
    static bool isVRAvailable();

    /**
     * @brief Renders both views onto a screen quad
     */
    static void renderToScreen();

    /**
     * @brief Get the role of a specific device
     *
     * @param device_index The device index
     *
     * @return The role of this index
     */
    static Role getDeviceRole(const uint32_t device_index);

    /**
     * @brief Get the device pose transform from device to world space.
     *
     * @param device_index The device index
     *
     * @return The transformation matrix
     */
    static glm::mat4 getDevicePose(const uint32_t device_index);

    /**
     * @brief Set the current movement line
     *
     * @param start The start point
     * @param end The end point
     */
    static void setMovementLine(const glm::vec3& start, const glm::vec3& end);

    /**
     * @brief Draw the current movement line
     *
     * @param camera The camera
     */
    static void drawMovementLine(const atcg::ref_ptr<atcg::PerspectiveCamera>& camera);

    /**
     * @brief Set a offset between the tracking frame and the virtual world
     *
     * @param offset The new offset
     */
    static void setOffset(const glm::vec3& offset);

    /**
     * @brief Get the offset between the tracking frame and the virtual world
     *
     * @return The offset
     */
    static glm::vec3 getOffset();

    /**
     * @brief Destroys the Renderer instance. SHOULD NOT BE CALLED BY THE USER
     *
     */
    inline static void destroy() { delete s_renderer; }

    ~VRRenderer();

private:
    VRRenderer();

    class Impl;
    atcg::scope_ptr<Impl> impl;

    static VRRenderer* s_renderer;
};
}    // namespace atcg