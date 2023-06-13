#include <Renderer/Context.h>

#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <Core/Log.h>

namespace atcg
{

namespace detail
{
void GLAPIENTRY MessageCallback(GLenum source,
                                GLenum type,
                                GLuint id,
                                GLenum severity,
                                GLsizei length,
                                const GLchar* message,
                                const void* userParam)
{
    switch(severity)
    {
        case GL_DEBUG_SEVERITY_LOW:
        case GL_DEBUG_SEVERITY_MEDIUM:
        {
            if(id == 131218) return;    // Some NVIDIA stuff going wrong -> disable this warning
            ATCG_WARN(message);
        }
        break;
        case GL_DEBUG_SEVERITY_HIGH:
        {
            ATCG_ERROR(message);
        }
        break;
        default:
            break;
    }
}
}    // namespace detail

void Context::init(void* window)
{
    glfwMakeContextCurrent((GLFWwindow*)window);

    if(!gladLoadGL()) { ATCG_ERROR("Error loading glad!"); }

    _window = window;

#ifndef NDEBUG
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(detail::MessageCallback, 0);
#endif
}

void Context::swapBuffers()
{
    glfwSwapBuffers((GLFWwindow*)_window);
}
}    // namespace atcg