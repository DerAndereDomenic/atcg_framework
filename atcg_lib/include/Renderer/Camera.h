#pragma once

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

    protected:
        virtual void recalculateView() = 0;
        virtual void recalculateProjection() = 0;

        glm::mat4 _view;
        glm::mat4 _projection;

    private:

    };
}