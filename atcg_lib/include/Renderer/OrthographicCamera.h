#pragma once

#include <Renderer/Camera.h>

namespace atcg
{
/**
 * @brief A class to model an orthographic camera
 *
 */
class OrthographicCamera : public Camera
{
public:
    /**
     * @brief Construct a new Orthographic Camera object
     *
     * @param left
     * @param right
     * @param bottom
     * @param top
     */
    OrthographicCamera(const float& left, const float& right, const float& bottom, const float& top);

    /**
     * @brief Destroy the Orthographic Camera object
     *
     */
    ~OrthographicCamera() = default;

    /**
     * @brief Get the Projection matrix
     *
     * @return glm::mat4 The projection matrix
     */
    ATCG_INLINE virtual glm::mat4 getProjection() const override { return _projection; }

    /**
     * @brief Get the View Projection matrix
     *
     * @return glm::mat4 The view-projection matrix
     */
    ATCG_INLINE virtual glm::mat4 getViewProjection() const override { return _projection; }

    /**
     * @brief Get the View matrix
     *
     * @return glm::mat4 The view matrix
     */
    ATCG_INLINE virtual glm::mat4 getView() const override { return glm::mat4(1); }

    /**
     * @brief Get the Position
     *
     * @return glm::vec3 The position
     */
    ATCG_INLINE virtual glm::vec3 getPosition() const override { return glm::vec3(0); }

    /**
     * @brief Get the Direction
     *
     * @return glm::vec3 The view direction
     */
    ATCG_INLINE virtual glm::vec3 getDirection() const override { return glm::vec3(0, 0, -1); }

    /**
     * @brief Set the Projection
     *
     * @param left
     * @param right
     * @param bottom
     * @param top
     */
    ATCG_INLINE void setProjection(const float& left, const float& right, const float& bottom, const float& top)
    {
        _left   = left;
        _right  = right;
        _bottom = bottom;
        _top    = top;
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
    float _left, _right, _bottom, _top;

    glm::mat4 _projection;
};
}    // namespace atcg