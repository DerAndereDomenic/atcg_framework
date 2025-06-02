#include <Core/Input.h>
#include <Core/Application.h>
#include <Core/Assert.h>

#ifndef ATCG_HEADLESS
    #include <GLFW/glfw3.h>
#endif

// Based on Hazel Engine (https://github.com/TheCherno/Hazel)
// Modified by Domenic Zingsheim in 2023

namespace atcg
{
bool Input::isKeyPressed(const int32_t& key)
{
#ifndef ATCG_HEADLESS
    ATCG_ASSERT(Application::get(), "There must be a valid application");
    ATCG_ASSERT(Application::get()->getWindow(), "There must be a window initialized");

    auto* window = Application::get()->getWindow()->getNativeWindow();
    auto state   = glfwGetKey((GLFWwindow*)window, key);
    return state == GLFW_PRESS;
#else
    return false;
#endif
}

bool Input::isKeyReleased(const int32_t& key)
{
#ifndef ATCG_HEADLESS
    ATCG_ASSERT(Application::get(), "There must be a valid application");
    ATCG_ASSERT(Application::get()->getWindow(), "There must be a window initialized");

    auto* window = Application::get()->getWindow()->getNativeWindow();
    auto state   = glfwGetKey((GLFWwindow*)window, key);
    return state == GLFW_RELEASE;
#else
    return false;
#endif
}

bool Input::isMouseButtonPressed(const int32_t& button)
{
#ifndef ATCG_HEADLESS
    ATCG_ASSERT(Application::get(), "There must be a valid application");
    ATCG_ASSERT(Application::get()->getWindow(), "There must be a window initialized");

    auto* window = Application::get()->getWindow()->getNativeWindow();
    auto state   = glfwGetMouseButton((GLFWwindow*)window, button);
    return state == GLFW_PRESS;
#else
    return false;
#endif
}

glm::vec2 Input::getMousePosition()
{
#ifndef ATCG_HEADLESS
    ATCG_ASSERT(Application::get(), "There must be a valid application");
    ATCG_ASSERT(Application::get()->getWindow(), "There must be a window initialized");

    auto* window = Application::get()->getWindow()->getNativeWindow();
    double xpos, ypos;
    glfwGetCursorPos((GLFWwindow*)window, &xpos, &ypos);

    return glm::vec2(xpos, ypos);
#else
    return glm::vec2(0);
#endif
}
}    // namespace atcg