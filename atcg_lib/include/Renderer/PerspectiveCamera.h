#pragma once

#include <Core/glm.h>

#include <Renderer/Camera.h>

namespace atcg
{
/**
 * @brief A class to model a camera
 */
class PerspectiveCamera : public Camera
{
public:
    /**
     * @brief Construct a new Camera object
     *
     * @param extrinsics The camera extrinsics
     * @param intrinsics The camera intrinsics
     */
    PerspectiveCamera(const CameraExtrinsics& extrinsics = {}, const CameraIntrinsics& intrinsics = {});

    /**
     * @brief Get the Position
     *
     * @return glm::vec3 The position
     */
    ATCG_INLINE virtual glm::vec3 getPosition() const override { return _extrinsics.position(); }

    /**
     * @brief Get the Direction
     *
     * @return glm::vec3 The view direction
     */
    ATCG_INLINE virtual glm::vec3 getDirection() const override
    {
        return glm::normalize(_extrinsics.position() - _extrinsics.target());
    }

    /**
     * @brief Get the Look At target
     *
     * @return glm::vec3 The look at target
     */
    ATCG_INLINE glm::vec3 getLookAt() const { return _extrinsics.target(); }

    /**
     * @brief Get the Up direction
     *
     * @return glm::vec3 The up direction
     */
    ATCG_INLINE glm::vec3 getUp() const { return glm::vec3(0, 1, 0); }

    /**
     * @brief Get the Projection matrix
     *
     * @return glm::mat4 The projection matrix
     */
    ATCG_INLINE virtual glm::mat4 getProjection() const override { return _intrinsics.projection(); }

    /**
     *  @brief Set the projection matrix
     *
     *  @param projection The new projection matrix
     */
    ATCG_INLINE void setProjection(const glm::mat4& projection) { _intrinsics.setProjection(projection); }

    /**
     * @brief Get the View Projection matrix
     *
     * @return glm::mat4 The view-projection matrix
     */
    ATCG_INLINE virtual glm::mat4 getViewProjection() const override
    {
        return _intrinsics.projection() * _extrinsics.extrinsicMatrix();
    }

    /**
     * @brief Get the Aspect Ratio
     *
     * @return float The aspect ratio
     */
    ATCG_INLINE float getAspectRatio() const { return _intrinsics.aspectRatio(); }

    /**
     * @brief Get the View matrix
     *
     * @return glm::mat4 The view matrix
     */
    ATCG_INLINE virtual glm::mat4 getView() const override { return _extrinsics.extrinsicMatrix(); }

    /**
     *  @brief Set the view matrix
     *
     *  @param view The new orthonormal view matrix
     */
    ATCG_INLINE void setView(const glm::mat4& view) { _extrinsics.setExtrinsicMatrix(view); }

    /**
     * @brief Set the Position
     *
     * @param position The new position
     */
    ATCG_INLINE void setPosition(const glm::vec3& position) { _extrinsics.setPosition(position); }

    /**
     * @brief Set the Look At Target
     *
     * @param look_at The new target
     */
    ATCG_INLINE void setLookAt(const glm::vec3& look_at) { _extrinsics.setTarget(look_at); }

    /**
     * @brief Set the Aspect Ratio
     *
     * @param aspect_ratio The new aspect ratio
     */
    ATCG_INLINE void setAspectRatio(const float& aspect_ratio) { _intrinsics.setAspectRatio(aspect_ratio); }

    /**
     * @brief Set the fov in y direction
     *
     * @param fov The fov in degrees
     */
    ATCG_INLINE void setFOV(const float& fov) { _intrinsics.setFOV(fov); }

    /**
     * @brief Get the near plane
     *
     * @return The near plane
     */
    ATCG_INLINE float getNear() const { return _intrinsics.zNear(); }

    /**
     * @brief Get the far plane
     *
     * @return The far plane
     */
    ATCG_INLINE float getFar() const { return _intrinsics.zFar(); }

    /**
     * @brief Get the camera fov in y direction
     *
     * @return The fov (in degree)
     */
    ATCG_INLINE float getFOV() const { return _intrinsics.FOV(); }

    /**
     * @brief Set the near plane
     *
     * @param near_plane The near plane
     */
    ATCG_INLINE void setNear(float near_plane) { _intrinsics.setNear(near_plane); }

    /**
     * @brief Set the far plane
     *
     * @param far_plane The far plane
     */
    ATCG_INLINE void setFar(float far_plane) { _intrinsics.setFar(far_plane); }

    /**
     * @brief Create a copy of the camera
     *
     * @return The deep copy
     */
    virtual atcg::ref_ptr<Camera> copy() const override;
};
}    // namespace atcg