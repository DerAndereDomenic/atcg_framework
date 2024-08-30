#include <Renderer/Context.h>

#include <glad/glad.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
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

static EGLDisplay eglDisplay;
static EGLContext eglContext;
static EGLSurface eglSurface;

}    // namespace detail

void Context::destroy()
{
    if(_context_handle)
    {
        deactivate();
    }
}

void Context::create()
{
    bool api = eglBindAPI(EGL_OPENGL_API);
    ATCG_ASSERT(api, "Could not bind opengl API");

    detail::eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    ATCG_ASSERT(detail::eglDisplay != EGL_NO_DISPLAY, "Could not create EGL display");

    bool initialiazed = eglInitialize(detail::eglDisplay, nullptr, nullptr);
    ATCG_ASSERT(initialiazed, "Could not initialize EGL");

    // Choose EGL config
    EGLint configAttribs[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE};
    EGLConfig eglConfig;
    EGLint numConfigs;
    bool choose = eglChooseConfig(detail::eglDisplay, configAttribs, &eglConfig, 1, &numConfigs);
    ATCG_ASSERT(choose, "Failed to choose display");

    // Create an EGL context
    EGLint contextAttribs[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 6, EGL_NONE};
    detail::eglContext      = eglCreateContext(detail::eglDisplay, eglConfig, EGL_NO_CONTEXT, contextAttribs);
    ATCG_ASSERT(detail::eglContext != EGL_NO_CONTEXT, "Failed to create context");

    // Create an EGL window surface from the GLFW window
    EGLint pbufferAttribs[] = {EGL_WIDTH,
                               1024,
                               EGL_HEIGHT,
                               1024,
                               EGL_TEXTURE_FORMAT,
                               EGL_TEXTURE_RGBA,
                               EGL_TEXTURE_TARGET,
                               EGL_TEXTURE_2D,
                               EGL_MIPMAP_TEXTURE,
                               EGL_TRUE,
                               EGL_NONE};    // TODO
    detail::eglSurface      = eglCreatePbufferSurface(detail::eglDisplay, eglConfig, pbufferAttribs);
    // detail::eglSurface =
    //     eglCreateWindowSurface(detail::eglDisplay, eglConfig, glfwGetX11Window((GLFWwindow*)window), nullptr);
    // ATCG_ASSERT(detail::eglSurface != EGL_NO_SURFACE, "Failed to create surface");

    // Make the context current
    eglMakeCurrent(detail::eglDisplay, detail::eglSurface, detail::eglSurface, detail::eglContext);

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
    eglSwapBuffers(detail::eglDisplay, detail::eglSurface);
}

void Context::makeCurrent()
{
    eglMakeCurrent(detail::eglDisplay, detail::eglSurface, detail::eglSurface, detail::eglContext);
}

void Context::deactivate()
{
    eglMakeCurrent(detail::eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
}
}    // namespace atcg