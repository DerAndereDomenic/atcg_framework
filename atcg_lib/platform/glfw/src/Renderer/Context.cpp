#include <Renderer/Context.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <Core/Assert.h>

namespace atcg
{

namespace detail
{
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

void Context::destroy()
{
    if(_context_handle)
    {
        deactivate();
        glfwDestroyWindow((GLFWwindow*)_context_handle);
    }
}

void Context::create()
{
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    _context_handle = (void*)glfwCreateWindow(1, 1, "ATCG", nullptr, nullptr);
    ATCG_ASSERT(_context_handle, "Could not create context");
    makeCurrent();
}

void Context::initGraphicsAPI()
{
    if(!detail::s_opengl_initialized)
    {
        makeCurrent();

        if(!gladLoadGL())
        {
            ATCG_ERROR("Error loading glad!");
        }

#ifndef NDEBUG
        glEnable(GL_DEBUG_OUTPUT);
        glDebugMessageCallback(detail::MessageCallback, 0);
#endif
        detail::s_opengl_initialized = true;
    }
}

void Context::swapBuffers()
{
    glfwSwapBuffers((GLFWwindow*)_context_handle);
}

void Context::makeCurrent()
{
    glfwMakeContextCurrent((GLFWwindow*)_context_handle);
}

void Context::deactivate()
{
    glfwMakeContextCurrent(NULL);
}
}    // namespace atcg