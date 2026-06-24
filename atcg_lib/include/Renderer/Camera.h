#pragma once

#include <Core/API.h>
#include <Core/glm.h>

#include <Renderer/CameraUtils.h>

namespace atcg
{
class ATCG_API Camera
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

    /**
     * @brief Transform a point from world space to camera space
     *
     * @param point The point in world space
     * @return The point in camera space
     */
    virtual glm::vec3 transformToCameraSpace(const glm::vec3& point) const = 0;

    /**
     * @brief Transform a point from world space to normalized device coordinates (NDC)
     *
     * @param point The point in world space
     * @return The point in NDC
     */
    virtual glm::vec3 transformToNormalizedDeviceCoordinates(const glm::vec3& point) const = 0;

    /**
     * @brief Check if a point is inside the camera frustum
     *
     * @param point The point in world space
     * @return true if the point is inside the frustum, false otherwise
     */
    virtual bool isPointInFrustum(const glm::vec3& point) const = 0;

    ATCG_INLINE const CameraExtrinsics& getExtrinsics() const { return _extrinsics; }

    ATCG_INLINE const CameraIntrinsics& getIntrinsics() const { return _intrinsics; }

    ATCG_INLINE void setIntrinsics(const CameraIntrinsics& intrinsics) { _intrinsics = intrinsics; }

    ATCG_INLINE void setExtrinsics(const CameraExtrinsics& extrinsics) { _extrinsics = extrinsics; }

protected:
    virtual void recalculateView() {};
    virtual void recalculateProjection() {};

    CameraExtrinsics _extrinsics;
    CameraIntrinsics _intrinsics;

private:
};
}    // namespace atcg