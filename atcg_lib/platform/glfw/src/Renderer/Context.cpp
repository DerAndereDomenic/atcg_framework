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

void Context::create(const int device_id)
{
    ATCG_ASSERT(!_context_handle, "Context already created");

    if(device_id != 0)
    {
        ATCG_WARN("A device id that is not 0 ({0}) was requested for glfw backend. This is unsupported", device_id);
    }

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    _context_handle = (void*)glfwCreateWindow(1, 1, "ATCG", nullptr, nullptr);
    ATCG_ASSERT(_context_handle, "Could not create context");
    makeCurrent();
}

void Context::create(const atcg::ref_ptr<Context>& shared)
{
    ATCG_ASSERT(!_context_handle, "Context already created");

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* glfwShared = nullptr;

    if(shared)
    {
        shared->deactivate();
        glfwShared = (GLFWwindow*)shared->getContextHandle();
    }

    _context_handle = (void*)glfwCreateWindow(1, 1, "ATCG", nullptr, glfwShared);
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