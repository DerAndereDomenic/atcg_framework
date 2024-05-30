#include <Core/Input.h>
#include <Core/Application.h>
#include <GLFW/glfw3.h>

namespace atcg
{
bool Input::isKeyPressed(const int32_t& key)
{
    auto* window = Application::get()->getWindow()->getNativeWindow();
    auto state   = glfwGetKey((GLFWwindow*)window, key);
    return state == GLFW_PRESS;
}

bool Input::isKeyReleased(const int32_t& key)
{
    auto* window = Application::get()->getWindow()->getNativeWindow();
    auto state   = glfwGetKey((GLFWwindow*)window, key);
    return state == GLFW_RELEASE;
}

bool Input::isMouseButtonPressed(const int32_t& button)
{
    auto* window = Application::get()->getWindow()->getNativeWindow();
    auto state   = glfwGetMouseButton((GLFWwindow*)window, button);
    return state == GLFW_PRESS;
}

glm::vec2 Input::getMousePosition()
{
    auto* window = Application::get()->getWindow()->getNativeWindow();
    double xpos, ypos;
    glfwGetCursorPos((GLFWwindow*)window, &xpos, &ypos);

    return glm::vec2(xpos, ypos);
}
}    // namespace atcg