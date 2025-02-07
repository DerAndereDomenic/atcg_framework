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
     * @param aspect_ratio The aspect ratio
     * @param position The camera position
     * @param look_at The camera's look at target
     */
    PerspectiveCamera(const float& aspect_ratio,
                      const glm::vec3& position = glm::vec3(0, 0, -1),
                      const glm::vec3& look_at  = glm::vec3(0));

    /**
     * @brief Get the Position
     *
     * @return glm::vec3 The position
     */
    ATCG_INLINE virtual glm::vec3 getPosition() const override { return _position; }

    /**
     * @brief Get the Direction
     *
     * @return glm::vec3 The view direction
     */
    ATCG_INLINE virtual glm::vec3 getDirection() const override { return glm::normalize(_position - _look_at); }

    /**
     * @brief Get the Look At target
     *
     * @return glm::vec3 The look at target
     */
    ATCG_INLINE glm::vec3 getLookAt() const { return _look_at; }

    /**
     * @brief Get the Up direction
     *
     * @return glm::vec3 The up direction
     */
    ATCG_INLINE glm::vec3 getUp() const { return _up; }

    /**
     * @brief Get the Projection matrix
     *
     * @return glm::mat4 The projection matrix
     */
    ATCG_INLINE virtual glm::mat4 getProjection() const override { return _projection; }

    /**
     *  @brief Set the projection matrix
     *
     *  @param projection The new projection matrix
     */
    void setProjection(const glm::mat4& projection);

    /**
     * @brief Get the View Projection matrix
     *
     * @return glm::mat4 The view-projection matrix
     */
    ATCG_INLINE virtual glm::mat4 getViewProjection() const override { return _projection * _view; }

    /**
     * @brief Get the Aspect Ratio
     *
     * @return float The aspect ratio
     */
    ATCG_INLINE float getAspectRatio() const { return _aspect_ratio; }

    /**
     * @brief Get the View matrix
     *
     * @return glm::mat4 The view matrix
     */
    ATCG_INLINE virtual glm::mat4 getView() const override { return _view; }

    /**
     *  @brief Set the view matrix
     *
     *  @param view The new orthonormal view matrix
     */
    void setView(const glm::mat4& view);

    /**
     * @brief Set view and projection from transform
     *
     * @param model The model matrix
     */
    void setFromTransform(const glm::mat4& transform);

    /**
     * @brief Convert the camera representation into a transform (object to world transform)
     *
     * @return The transform
     */
    glm::mat4 getAsTransform() const;

    /**
     * @brief Set the Position
     *
     * @param position The new position
     */
    ATCG_INLINE void setPosition(const glm::vec3& position)
    {
        _position = position;
        recalculateView();
    }

    /**
     * @brief Set the Look At Target
     *
     * @param look_at The new target
     */
    ATCG_INLINE void setLookAt(const glm::vec3& look_at)
    {
        _look_at = look_at;
        recalculateView();
    }

    /**
     * @brief Set the Aspect Ratio
     *
     * @param aspect_ratio The new aspect ratio
     */
    ATCG_INLINE void setAspectRatio(const float& aspect_ratio)
    {
        _aspect_ratio = aspect_ratio;
        recalculateProjection();
    }

    ATCG_INLINE void setFOV(const float& fov)
    {
        _fovy = fov;
        recalculateProjection();
    }

    /**
     * @brief Get the near plane
     *
     * @return The near plane
     */
    ATCG_INLINE float getNear() const { return _near; }

    /**
     * @brief Get the far plane
     *
     * @return The far plane
     */
    ATCG_INLINE float getFar() const { return _far; }

    /**
     * @brief Get the camera fov in y direction
     *
     * @return The fov (in degree)
     */
    ATCG_INLINE float getFOV() const { return _fovy; }

    /**
     * @brief Set the near plane
     *
     * @param near_plane The near plane
     */
    ATCG_INLINE void setNear(float near_plane)
    {
        _near = near_plane;
        recalculateProjection();
    }

    /**
     * @brief Set the far plane
     *
     * @param far_plane The far plane
     */
    ATCG_INLINE void setFar(float far_plane)
    {
        _far = far_plane;
        recalculateProjection();
    }

    /**
     * @brief Create a copy of the camera
     *
     * @return The deep copy
     */
    virtual atcg::ref_ptr<Camera> copy() const override;

protected:
    virtual void recalculateView() override;
    virtual void recalculateProjection() override;

private:
    // View parameters
    glm::vec3 _position;
    glm::vec3 _up;
    glm::vec3 _look_at;

    // Projection parameters
    glm::vec2 _optical_center;
    float _aspect_ratio;
    float _fovy;
    float _near;
    float _far;
};
}    // namespace atcg