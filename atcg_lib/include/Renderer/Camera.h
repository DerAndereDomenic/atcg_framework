#pragma once

#include <glm/glm.hpp>

namespace atcg
{
    /**
     * @brief A class to model a camera
     */
    class Camera
    {
    public:

        /**
         * @brief Construct a new Camera object
         * 
         * @param aspect_ratio The aspect ratio
         * @param position The camera position
         * @param look_at The camera's look at target
         */
        Camera(const float& aspect_ratio, const glm::vec3& position = glm::vec3(0), const glm::vec3& look_at = glm::vec3(0));

        /**
         * @brief Get the Position
         * 
         * @return glm::vec3 The position
         */
        inline glm::vec3 getPosition() const {return _position;}

        /**
         * @brief Get the Look At target
         * 
         * @return glm::vec3 The look at target
         */
        inline glm::vec3 getLookAt() const {return _look_at;}

        /**
         * @brief Get the Up direction
         * 
         * @return glm::vec3 The up direction
         */
        inline glm::vec3 getUp() const {return _up;}
        
        /**
         * @brief Get the Projection matrix
         * 
         * @return glm::mat4 The projection matrix
         */
        inline glm::mat4 getProjection() const {return _projection;}

        /**
         * @brief Get the View Projection matrix
         * 
         * @return glm::mat4 The view-projection matrix
         */
        inline glm::mat4 getViewProjection() const {return _projection * _view;}

        /**
         * @brief Get the Aspect Ratio
         * 
         * @return float The aspect ratio
         */
        inline float getAspectRatio() const {return _aspect_ratio;}

        /**
         * @brief Get the View matrix
         * 
         * @return glm::mat4 The view matrix
         */
        inline glm::mat4 getView() const {return _view;}

        /**
         * @brief Set the Position
         * 
         * @param position The new position
         */
        inline void setPosition(const glm::vec3& position) {_position = position; recalculateView();}

        /**
         * @brief Set the Look At Target
         * 
         * @param look_at The new target
         */
        inline void setLookAt(const glm::vec3& look_at) {_look_at = look_at; recalculateView();}

        /**
         * @brief Set the Aspect Ratio
         * 
         * @param aspect_ratio The new aspect ratio
         */
        inline void setAspectRatio(const float& aspect_ratio) {_aspect_ratio = aspect_ratio; recalculateProjection();}

    private:
        void recalculateView();
        void recalculateProjection();

    private:
        glm::vec3 _position;
        glm::vec3 _up;
        glm::vec3 _look_at;

        float _aspect_ratio;

        glm::mat4 _view;
        glm::mat4 _projection;
    };
}