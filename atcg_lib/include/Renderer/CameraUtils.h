#pragma once

#include <Core/glm.h>

namespace atcg
{
/**
 * @brief A class to model camera extrinsics
 */
class CameraExtrinsics
{
public:
    /**
     * @brief Default constructor
     */
    CameraExtrinsics() = default;

    /**
     * @brief Constructor
     *
     * @param position The position
     * @param target The look at target
     */
    CameraExtrinsics(const glm::vec3& position, const glm::vec3& target);

    /**
     * @brief Constructor
     *
     * @param extrinsics_matrix The camera extrinsic matrix
     */
    CameraExtrinsics(const glm::mat4& extrinsics_matrix);

    /**
     * @brief Set the position
     *
     * @param position
     */
    void setPosition(const glm::vec3& position);

    /**
     * @brief Set the look at target
     *
     * @param target
     */
    void setTarget(const glm::vec3& target);

    /**
     * @brief Set extrinsic matrix
     *
     * @param extrinsics The extrinsic matrix
     */
    void setExtrinsicMatrix(const glm::mat4& extrinsics);

    /**
     * @brief Get the position
     *
     * @return The position
     */
    ATCG_INLINE const glm::vec3& position() const { return _position; }

    /**
     * @brief Get the target
     *
     * @return The look at target
     */
    ATCG_INLINE const glm::vec3& target() const { return _target; }

    /**
     * @brief Get the extrinsics matrix
     *
     * @return The extrinsic matrix
     */
    ATCG_INLINE const glm::mat4& extrinsicMatrix() const { return _extrinsics_matrix; }

private:
    void recalculateView();

private:
    glm::vec3 _position = glm::vec3(0);
    glm::vec3 _target   = glm::vec3(0, 0, -1);

    glm::mat4 _extrinsics_matrix = glm::mat4(1);
};

/**
 * @brief A class to model camera intrinsics
 */
class CameraIntrinsics
{
public:
    /**
     * @brief Default constructor
     */
    CameraIntrinsics() = default;

    /**
     * @brief Constructor
     *
     * @param aspect_ratio The aspect ratio
     * @param fov_y The fov in y direction
     * @param znear The near plane
     * @param zfar The far plane
     * @param optical_center The optical center
     */
    CameraIntrinsics(const float aspect_ratio,
                     const float fov_y,
                     const float znear               = 0.01f,
                     const float zfar                = 1000.0f,
                     const glm::vec2& optical_center = glm::vec2(0));

    /**
     * @brief Constructor
     *
     * @param projection The projection matrix
     */
    CameraIntrinsics(const glm::mat4& projection);

    /**
     * @brief Set the aspect ratio
     *
     * @param aspect_ratio
     */
    void setAspectRatio(const float aspect_ratio);

    /**
     * @brief Set the y FOV
     *
     * @param fov_y The fov in degrees
     */
    void setFOV(const float fov_y);

    /**
     * @brief Set the near plane
     *
     * @param znear The near plane
     */
    void setNear(const float znear);

    /**
     * @brief Set the far plane
     *
     * @param zfar The far plane
     */
    void setFar(const float zfar);

    /**
     * @brief Set the optical center
     *
     * @param optical_center The optical center
     */
    void setOpticalCenter(const glm::vec2& optical_center);

    /**
     * @brief Set the projection matrix
     *
     * @param projection The projection matrix
     */
    void setProjection(const glm::mat4& projection);

    /**
     * @brief Get the aspect ratio
     *
     * @return The aspect ratio
     */
    ATCG_INLINE float aspectRatio() const { return _aspect_ratio; }

    /**
     * @brief Get the fov in y direction
     *
     * @return The fov in degrees
     */
    ATCG_INLINE float FOV() const { return _fov_y; }

    /**
     * @brief Get the near plane
     *
     * @return The near plane
     */
    ATCG_INLINE float zNear() const { return _near; }

    /**
     * @brief Get the far plane
     *
     * @return The far plane
     */
    ATCG_INLINE float zFar() const { return _far; }

    /**
     * @brief Get the optical center
     *
     * @return The optical center
     */
    ATCG_INLINE const glm::vec2& opticalCenter() const { return _optical_center; }

    /**
     * @brief Get the projection matrix
     *
     * @return The projection matrix
     */
    ATCG_INLINE const glm::mat4& projection() const { return _projection; }

private:
    void recalculateProjection();

private:
    float _aspect_ratio       = 1.0f;
    float _fov_y              = 60.0f;
    float _near               = 0.01f;
    float _far                = 1000.0f;
    glm::vec2 _optical_center = glm::vec2(0);

    glm::mat4 _projection = glm::perspective(glm::radians(_fov_y), _aspect_ratio, _near, _far);
};
}    // namespace atcg