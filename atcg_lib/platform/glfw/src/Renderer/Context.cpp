#include <Renderer/Context.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

namespace atcg
{

namespace detail
{
static bool s_glfw_initialized   = false;
static bool s_opengl_initialized = false;

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


static void GLFWErrorCallback(int error, const char* description)
{
    ATCG_ERROR("GLFW Error: {0}: {1}", error, description);
}

}    // namespace detail

void Context::initWindowingAPI()
{
    if(!detail::s_glfw_initialized)
    {
        int success = glfwInit();
        if(success != GLFW_TRUE) return;

        detail::s_glfw_initialized = true;
        glfwSetErrorCallback(detail::GLFWErrorCallback);
    }
}

void Context::deinitWindowingAPI()
{
    if(detail::s_glfw_initialized)
    {
        glfwTerminate();
        detail::s_glfw_initialized = false;
    }
}

void Context::initGraphicsAPI(void* window)
{
    if(!detail::s_opengl_initialized)
    {
        glfwMakeContextCurrent((GLFWwindow*)window);

        if(!gladLoadGL())
        {
            ATCG_ERROR("Error loading glad!");
        }

        _window = window;

#ifndef NDEBUG
        glEnable(GL_DEBUG_OUTPUT);
        glDebugMessageCallback(detail::MessageCallback, 0);
#endif
        detail::s_opengl_initialized = true;
    }
}

void Context::swapBuffers()
{
    glfwSwapBuffers((GLFWwindow*)_window);
}

void Context::makeCurrent()
{
    glfwMakeContextCurrent((GLFWwindow*)_window);
}

void Context::deactivate()
{
    glfwMakeContextCurrent(NULL);
}
}    // namespace atcg