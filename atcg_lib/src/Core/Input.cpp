#include <Core/Input.h>
#include <Core/Application.h>
#include <Core/Assert.h>
#include <GLFW/glfw3.h>

namespace atcg
{
bool Input::isKeyPressed(const int32_t& key)
{
    ATCG_ASSERT(Application::get(), "There must be a valid application");
    ATCG_ASSERT(Application::get()->getWindow(), "There must be a window initialized");

    auto* window = Application::get()->getWindow()->getNativeWindow();
    auto state   = glfwGetKey((GLFWwindow*)window, key);
    return state == GLFW_PRESS;
}

bool Input::isKeyReleased(const int32_t& key)
{
    ATCG_ASSERT(Application::get(), "There must be a valid application");
    ATCG_ASSERT(Application::get()->getWindow(), "There must be a window initialized");

    auto* window = Application::get()->getWindow()->getNativeWindow();
    auto state   = glfwGetKey((GLFWwindow*)window, key);
    return state == GLFW_RELEASE;
}

bool Input::isMouseButtonPressed(const int32_t& button)
{
    ATCG_ASSERT(Application::get(), "There must be a valid application");
    ATCG_ASSERT(Application::get()->getWindow(), "There must be a window initialized");

    auto* window = Application::get()->getWindow()->getNativeWindow();
    auto state   = glfwGetMouseButton((GLFWwindow*)window, button);
    return state == GLFW_PRESS;
}

glm::vec2 Input::getMousePosition()
{
    ATCG_ASSERT(Application::get(), "There must be a valid application");
    ATCG_ASSERT(Application::get()->getWindow(), "There must be a window initialized");

    auto* window = Application::get()->getWindow()->getNativeWindow();
    double xpos, ypos;
    glfwGetCursorPos((GLFWwindow*)window, &xpos, &ypos);

    return glm::vec2(xpos, ypos);
}
}    // namespace atcg