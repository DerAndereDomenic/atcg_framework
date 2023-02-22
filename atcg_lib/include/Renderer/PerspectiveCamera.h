#pragma once

#include <glm/glm.hpp>

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
                      const glm::vec3& position = glm::vec3(0),
                      const glm::vec3& look_at  = glm::vec3(0));

    /**
     * @brief Get the Position
     *
     * @return glm::vec3 The position
     */
    inline virtual glm::vec3 getPosition() const override { return _position; }

    /**
     * @brief Get the Direction
     *
     * @return glm::vec3 The view direction
     */
    inline virtual glm::vec3 getDirection() const override { return glm::normalize(_position - _look_at); }

    /**
     * @brief Get the Look At target
     *
     * @return glm::vec3 The look at target
     */
    inline glm::vec3 getLookAt() const { return _look_at; }

    /**
     * @brief Get the Up direction
     *
     * @return glm::vec3 The up direction
     */
    inline glm::vec3 getUp() const { return _up; }

    /**
     * @brief Get the Projection matrix
     *
     * @return glm::mat4 The projection matrix
     */
    inline virtual glm::mat4 getProjection() const override { return _projection; }

    /**
     * @brief Get the View Projection matrix
     *
     * @return glm::mat4 The view-projection matrix
     */
    inline virtual glm::mat4 getViewProjection() const override { return _projection * _view; }

    /**
     * @brief Get the Aspect Ratio
     *
     * @return float The aspect ratio
     */
    inline float getAspectRatio() const { return _aspect_ratio; }

    /**
     * @brief Get the View matrix
     *
     * @return glm::mat4 The view matrix
     */
    inline virtual glm::mat4 getView() const override { return _view; }

    /**
     *  @brief Set the view matrix
     *
     *  @param view The new view matrix
     */
    void setView(const glm::mat4& view);

    /**
     *  @brief Set the projection matrix
     *
     *  @param projection The new projection matrix
     */
    inline void setProjection(const glm::mat4& projection) { _projection = projection; }

    /**
     * @brief Set the Position
     *
     * @param position The new position
     */
    inline void setPosition(const glm::vec3& position)
    {
        _position = position;
        recalculateView();
    }

    /**
     * @brief Set the Look At Target
     *
     * @param look_at The new target
     */
    inline void setLookAt(const glm::vec3& look_at)
    {
        _look_at = look_at;
        recalculateView();
    }

    /**
     * @brief Set the Aspect Ratio
     *
     * @param aspect_ratio The new aspect ratio
     */
    inline void setAspectRatio(const float& aspect_ratio)
    {
        _aspect_ratio = aspect_ratio;
        recalculateProjection();
    }

protected:
    virtual void recalculateView() override;
    virtual void recalculateProjection() override;

private:
    glm::vec3 _position;
    glm::vec3 _up;
    glm::vec3 _look_at;

    float _aspect_ratio;
};
}    // namespace atcg