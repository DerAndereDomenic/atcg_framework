#pragma once

#include <Core/glm.h>

namespace atcg
{
struct TransformComponent
{
    /**
     * @brief Create a transform component
     *
     * @param position The position
     * @param scale The scale
     * @param rotation Rotation in Euler angles
     */
    TransformComponent(const glm::vec3& position = glm::vec3(0),
                       const glm::vec3& scale    = glm::vec3(1),
                       const glm::vec3& rotation = glm::vec3(0))
        : _position(position),
          _scale(scale),
          _rotation(rotation)
    {
        calculateModelMatrix();
    }

    /**
     * @brief Create a transform component
     *
     * @param model The model matrix
     */
    TransformComponent(const glm::mat4& model) : _model_matrix(model) { decomposeModelMatrix(); }

    /**
     * @brief Set the position
     *
     * @param position The position
     */
    ATCG_INLINE void setPosition(const glm::vec3& position)
    {
        _position = position;
        ++_revision;
        calculateModelMatrix();
    }

    /**
     * @brief Set the rotation (Euler angles)
     *
     * @param rotation The euler angles
     */
    ATCG_INLINE void setRotation(const glm::vec3& rotation)
    {
        _rotation = rotation;
        ++_revision;
        calculateModelMatrix();
    }

    /**
     * @brief Set the scale
     *
     * @param scale The scale
     */
    ATCG_INLINE void setScale(const glm::vec3& scale)
    {
        _scale = scale;
        ++_revision;
        calculateModelMatrix();
    }

    /**
     * @brief Set the model matrix
     *
     * @param model The model matrix
     */
    ATCG_INLINE void setModel(const glm::mat4& model)
    {
        _model_matrix = model;
        ++_revision;
        decomposeModelMatrix();
    }

    /**
     * @brief Get the model matrix
     * @return The model matrix
     */
    ATCG_INLINE glm::mat4 getModel() const { return _model_matrix; }

    /**
     * @brief Get the position
     * @return The position
     */
    ATCG_INLINE glm::vec3 getPosition() const { return _position; }

    /**
     * @brief Get the scale
     * @return The scale
     */
    ATCG_INLINE glm::vec3 getScale() const { return _scale; }

    /**
     * @brief Get the rotation
     * @return The rotation
     */
    ATCG_INLINE glm::vec3 getRotation() const { return _rotation; }

    /**
     * @brief Get the number of revision (edits) of the transform component
     * @return The number of revisions
     */
    ATCG_INLINE uint64_t revision() const { return _revision; }

    ATCG_INLINE operator glm::mat4() const { return _model_matrix; }

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Transform"; }

private:
    void calculateModelMatrix();
    void decomposeModelMatrix();

private:
    glm::mat4 _model_matrix = glm::mat4(1);
    glm::vec3 _position     = glm::vec3(0);
    glm::vec3 _scale        = glm::vec3(1);
    glm::vec3 _rotation     = glm::vec3(0);    // Euler Angles
    uint64_t _revision      = 0;
};
}    // namespace atcg