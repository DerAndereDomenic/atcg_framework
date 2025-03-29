#pragma once

#include <Core/glm.h>

#include <Renderer/CameraUtils.h>

namespace atcg
{
class Camera
{
public:
    /**
     * @brief Get the Projection matrix
     *
     * @return glm::mat4 The projection matrix
     */
    virtual glm::mat4 getProjection() const = 0;

    /**
     * @brief Get the View Projection matrix
     *
     * @return glm::mat4 The view-projection matrix
     */
    virtual glm::mat4 getViewProjection() const = 0;

    /**
     * @brief Get the View matrix
     *
     * @return glm::mat4 The view matrix
     */
    virtual glm::mat4 getView() const = 0;

    /**
     * @brief Get the Position
     *
     * @return glm::vec3 The position
     */
    virtual glm::vec3 getPosition() const = 0;

    /**
     * @brief Get the Direction
     *
     * @return glm::vec3 The view direction
     */
    virtual glm::vec3 getDirection() const = 0;

    /**
     * @brief Create a copy of the camera
     *
     * @return The deep copy
     */
    virtual atcg::ref_ptr<Camera> copy() const = 0;

    ATCG_INLINE const CameraExtrinsics& getExtrinsics() const { return _extrinsics; }

    ATCG_INLINE const CameraIntrinsics& getIntrinsics() const { return _intrinsics; }

protected:
    virtual void recalculateView()       {};
    virtual void recalculateProjection() {};

    CameraExtrinsics _extrinsics;
    CameraIntrinsics _intrinsics;

private:
};
}    // namespace atcg