#include <Core/Input.h>
#include <Core/Application.h>
#include <GLFW/glfw3.h>

namespace atcg
{
    bool Input::isKeyPressed(const int32_t& key)
    {
        auto* window = Application::get()->getWindow()->getNativeWindow();
        auto state = glfwGetKey(window, key);
        return state == GLFW_PRESS;
    }

    bool Input::isMouseButtonPressed(const int32_t& button)
    {
        auto* window = Application::get()->getWindow()->getNativeWindow();
        auto state = glfwGetMouseButton(window, button);
        return state == GLFW_PRESS;
    }

    glm::vec2 Input::getMousePosition()
    {
        auto* window = Application::get()->getWindow()->getNativeWindow();
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);

        return glm::vec2(xpos, ypos);
    }
}