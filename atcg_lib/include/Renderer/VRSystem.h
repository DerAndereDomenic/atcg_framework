#pragma once

#include <Core/Memory.h>
#include <Core/glm.h>
#include <Renderer/Framebuffer.h>
#include <Renderer/PerspectiveCamera.h>
#include <Scene/Scene.h>

#include <functional>

namespace atcg
{
class VRSystem
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
    void init(const EventCallbackFn& callback);

    /**
     * @brief Initialize controller renderings.
     * This has to be called by the client. Adds controller entities to the scene
     *
     * These entities get updated automatically based on the current pose of the tracker every frame.
     *
     * @param scene The scene
     */
    void initControllerMeshes(const atcg::ref_ptr<atcg::Scene>& scene);

    /**
     * @brief Gets called every frame and handles VR updates
     *
     * @param delta_time The delta frame time
     */
    void onUpdate(const float delta_time);

    /**
     * @brief Do the tracking. This is a blocking call and should be called at the beginning of the rendering thread.
     * This is done by the application, not the client.
     */
    void doTracking();

    /**
     * @brief This function pulls events from the headset and forwards them through the framework.
     *
     */
    void emitEvents();

    /**
     * @brief Get the inverse view matrix of a specific eye
     *
     * @param eye The eye
     *
     * @return The view matrix
     */
    glm::mat4 getInverseView(const Eye& eye);

    /**
     * @brief Get both inverse view matrices at the same time
     *
     * @return A tuple with the view matrices (Left, Right)
     */
    std::tuple<glm::mat4, glm::mat4> getInverseViews();

    /**
     * @brief Get the projection matrix of a specific eye
     *
     * @param eye The eye
     *
     * @return The projection matrix
     */
    glm::mat4 getProjection(const Eye& eye);

    /**
     * @brief Get borth projection matrices
     *
     * @return The projection matrices for both eyes (Left, Right)
     */
    std::tuple<glm::mat4, glm::mat4> getProjections();

    /**
     * @brief Get the rendertarget framebuffers of a specific eye
     *
     * @param eye The eye
     *
     * @return The render target
     */
    atcg::ref_ptr<Framebuffer> getRenderTarget(const Eye& eye);

    /**
     * @brief Get the render targets of both eyes
     *
     * @return A tuple with the framebuffers for each eye (Left, Right)
     */
    std::tuple<atcg::ref_ptr<Framebuffer>, atcg::ref_ptr<Framebuffer>> getRenderTargets();

    /**
     * @brief Get the position of the Headset in tracking space
     *
     * @return The 3D position in tracking space
     */
    glm::vec3 getPosition();

    /**
     * @brief Get the resolution of the Headset
     *
     * @return The width
     */
    uint32_t width();

    /**
     * @brief Get the resolution of the Headset
     *
     * @return The height
     */
    uint32_t height();

    /**
     * @brief Check if VR is available
     *
     * @return True if the HMD was found and the runtime could be initialized
     */
    bool isVRAvailable();

    /**
     * @brief Renders both views onto a screen quad
     */
    void renderToScreen();

    /**
     * @brief Get the role of a specific device
     *
     * @param device_index The device index
     *
     * @return The role of this index
     */
    Role getDeviceRole(const uint32_t device_index);

    /**
     * @brief Get the device pose transform from device to world space.
     *
     * @param device_index The device index
     *
     * @return The transformation matrix
     */
    glm::mat4 getDevicePose(const uint32_t device_index);

    /**
     * @brief Set the current movement line
     *
     * @param start The start point
     * @param end The end point
     */
    void setMovementLine(const glm::vec3& start, const glm::vec3& end);

    /**
     * @brief Draw the current movement line
     *
     * @param camera The camera
     */
    void drawMovementLine(const atcg::ref_ptr<atcg::PerspectiveCamera>& camera);

    /**
     * @brief Set a offset between the tracking frame and the virtual world
     *
     * @param offset The new offset
     */
    void setOffset(const glm::vec3& offset);

    /**
     * @brief Get the offset between the tracking frame and the virtual world
     *
     * @return The offset
     */
    glm::vec3 getOffset();

    /**
     * @brief Set the near plane of the projection matrices
     *
     * @param n The near plane
     */
    void setNear(const float n);

    /**
     * @brief Set the far plane of the projection matrices
     *
     * @param f The far plane
     */
    void setFar(const float f);

    VRSystem();

    ~VRSystem();

private:
    class Impl;
    atcg::scope_ptr<Impl> impl;
};

/**
 * @brief This namespace encapsulates the default VR system
 */
namespace VR
{
/**
 * @brief Initialize the VR renderer
 *
 * @param callback The callback function
 */
ATCG_INLINE void init(const VRSystem::EventCallbackFn& callback)
{
    SystemRegistry::instance()->getSystem<VRSystem>()->init(callback);
}

/**
 * @brief Initialize controller renderings.
 * This has to be called by the client. Adds controller entities to the scene
 *
 * These entities get updated automatically based on the current pose of the tracker every frame.
 *
 * @param scene The scene
 */
ATCG_INLINE void initControllerMeshes(const atcg::ref_ptr<atcg::Scene>& scene)
{
    SystemRegistry::instance()->getSystem<VRSystem>()->initControllerMeshes(scene);
}

/**
 * @brief Gets called every frame and handles VR updates
 *
 * @param delta_time The delta frame time
 */
ATCG_INLINE void onUpdate(const float delta_time)
{
    SystemRegistry::instance()->getSystem<VRSystem>()->onUpdate(delta_time);
}

/**
 * @brief Do the tracking. This is a blocking call and should be called at the beginning of the rendering thread.
 * This is done by the application, not the client.
 */
ATCG_INLINE void doTracking()
{
    SystemRegistry::instance()->getSystem<VRSystem>()->doTracking();
}

/**
 * @brief This function pulls events from the headset and forwards them through the framework.
 *
 */
ATCG_INLINE void emitEvents()
{
    SystemRegistry::instance()->getSystem<VRSystem>()->emitEvents();
}

/**
 * @brief Get the inverse view matrix of a specific eye
 *
 * @param eye The eye
 *
 * @return The view matrix
 */
ATCG_INLINE glm::mat4 getInverseView(const VRSystem::Eye& eye)
{
    return SystemRegistry::instance()->getSystem<VRSystem>()->getInverseView(eye);
}

/**
 * @brief Get both inverse view matrices at the same time
 *
 * @return A tuple with the view matrices (Left, Right)
 */
ATCG_INLINE std::tuple<glm::mat4, glm::mat4> getInverseViews()
{
    return SystemRegistry::instance()->getSystem<VRSystem>()->getInverseViews();
}

/**
 * @brief Get the projection matrix of a specific eye
 *
 * @param eye The eye
 *
 * @return The projection matrix
 */
ATCG_INLINE glm::mat4 getProjection(const VRSystem::Eye& eye)
{
    return SystemRegistry::instance()->getSystem<VRSystem>()->getProjection(eye);
}

/**
 * @brief Get borth projection matrices
 *
 * @return The projection matrices for both eyes (Left, Right)
 */
ATCG_INLINE std::tuple<glm::mat4, glm::mat4> getProjections()
{
    return SystemRegistry::instance()->getSystem<VRSystem>()->getProjections();
}

/**
 * @brief Get the rendertarget framebuffers of a specific eye
 *
 * @param eye The eye
 *
 * @return The render target
 */
ATCG_INLINE atcg::ref_ptr<Framebuffer> getRenderTarget(const VRSystem::Eye& eye)
{
    return SystemRegistry::instance()->getSystem<VRSystem>()->getRenderTarget(eye);
}

/**
 * @brief Get the render targets of both eyes
 *
 * @return A tuple with the framebuffers for each eye (Left, Right)
 */
ATCG_INLINE std::tuple<atcg::ref_ptr<Framebuffer>, atcg::ref_ptr<Framebuffer>> getRenderTargets()
{
    return SystemRegistry::instance()->getSystem<VRSystem>()->getRenderTargets();
}

/**
 * @brief Get the position of the Headset in tracking space
 *
 * @return The 3D position in tracking space
 */
ATCG_INLINE glm::vec3 getPosition()
{
    return SystemRegistry::instance()->getSystem<VRSystem>()->getPosition();
}

/**
 * @brief Get the resolution of the Headset
 *
 * @return The width
 */
ATCG_INLINE uint32_t width()
{
    return SystemRegistry::instance()->getSystem<VRSystem>()->width();
}

/**
 * @brief Get the resolution of the Headset
 *
 * @return The height
 */
ATCG_INLINE uint32_t height()
{
    return SystemRegistry::instance()->getSystem<VRSystem>()->height();
}

/**
 * @brief Check if VR is available
 *
 * @return True if the HMD was found and the runtime could be initialized
 */
ATCG_INLINE bool isVRAvailable()
{
    return SystemRegistry::instance()->getSystem<VRSystem>()->isVRAvailable();
}

/**
 * @brief Renders both views onto a screen quad
 */
ATCG_INLINE void renderToScreen()
{
    SystemRegistry::instance()->getSystem<VRSystem>()->renderToScreen();
}

/**
 * @brief Get the role of a specific device
 *
 * @param device_index The device index
 *
 * @return The role of this index
 */
ATCG_INLINE VRSystem::Role getDeviceRole(const uint32_t device_index)
{
    return SystemRegistry::instance()->getSystem<VRSystem>()->getDeviceRole(device_index);
}

/**
 * @brief Get the device pose transform from device to world space.
 *
 * @param device_index The device index
 *
 * @return The transformation matrix
 */
ATCG_INLINE glm::mat4 getDevicePose(const uint32_t device_index)
{
    return SystemRegistry::instance()->getSystem<VRSystem>()->getDevicePose(device_index);
}

/**
 * @brief Set the current movement line
 *
 * @param start The start point
 * @param end The end point
 */
ATCG_INLINE void setMovementLine(const glm::vec3& start, const glm::vec3& end)
{
    SystemRegistry::instance()->getSystem<VRSystem>()->setMovementLine(start, end);
}

/**
 * @brief Draw the current movement line
 *
 * @param camera The camera
 */
ATCG_INLINE void drawMovementLine(const atcg::ref_ptr<atcg::PerspectiveCamera>& camera)
{
    SystemRegistry::instance()->getSystem<VRSystem>()->drawMovementLine(camera);
}

/**
 * @brief Set a offset between the tracking frame and the virtual world
 *
 * @param offset The new offset
 */
ATCG_INLINE void setOffset(const glm::vec3& offset)
{
    SystemRegistry::instance()->getSystem<VRSystem>()->setOffset(offset);
}

/**
 * @brief Get the offset between the tracking frame and the virtual world
 *
 * @return The offset
 */
ATCG_INLINE glm::vec3 getOffset()
{
    return SystemRegistry::instance()->getSystem<VRSystem>()->getOffset();
}

/**
 * @brief Set the near plane of the projection matrices
 *
 * @param n The near plane
 */
ATCG_INLINE void setNear(const float n)
{
    SystemRegistry::instance()->getSystem<VRSystem>()->setNear(n);
}

/**
 * @brief Set the far plane of the projection matrices
 *
 * @param f The far plane
 */
ATCG_INLINE void setFar(const float f)
{
    SystemRegistry::instance()->getSystem<VRSystem>()->setFar(f);
}

}    // namespace VR

}    // namespace atcg