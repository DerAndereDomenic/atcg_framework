#pragma once

#include <Core/Memory.h>
#include <Core/glm.h>
#include <Renderer/Framebuffer.h>

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

    using EventCallbackFn = std::function<void(Event*)>;

    /**
     * @brief Initialize the VR renderer
     *
     * @param callback The callback function
     */
    static void init(const EventCallbackFn& callback);

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
     * @brief Get the position of the Headset
     *
     * @return The 3D position in world space
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