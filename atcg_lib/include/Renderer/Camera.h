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

        inline glm::vec3 getPosition() const {return _position;}

        inline glm::vec3 getLookAt() const {return _look_at;}

        inline glm::vec3 getUp() const {return _up;}
        
        inline glm::mat4 getProjection() const {return _projection;}

        inline glm::mat4 getViewProjection() const {return _projection * _view;}

        inline float getAspectRatio() const {return _aspect_ratio;}

        inline glm::mat4 getView() const {return _view;}

        inline void setPosition(const glm::vec3& position) {_position = position; recalculateView();}

        inline void setLookAt(const glm::vec3& look_at) {_look_at = look_at; recalculateView();}

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