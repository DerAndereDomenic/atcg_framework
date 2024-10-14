#include <Renderer/Context.h>

#include <glad/glad.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <Core/Assert.h>
#include <Renderer/ContextData.h>

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

static EGLDeviceEXT getDeviceFromIndex(const int device_id)
{
    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEX"
                                                                                                "T");

    PFNEGLQUERYDEVICEBINARYEXTPROC eglQueryDeviceBinaryEXT = (PFNEGLQUERYDEVICEBINARYEXTPROC)eglGetProcAddress("eglQuer"
                                                                                                               "yDevice"
                                                                                                               "BinaryE"
                                                                                                               "XT");

    // Query devices
    EGLint numDevices;
    bool query = eglQueryDevicesEXT(0, NULL, &numDevices);
    ATCG_ASSERT(query, "Could not query available devices");
    EGLDeviceEXT devices[numDevices];
    eglQueryDevicesEXT(numDevices, devices, &numDevices);

    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device_id));
    uint64_t query_uuid = *(uint64_t*)&deviceProp.uuid;

    EGLDeviceEXT selected_device;
    for(EGLint i = 0; i < numDevices; i++)
    {
        uint64_t uuid;
        int size;
        eglQueryDeviceBinaryEXT(devices[i], EGL_DEVICE_UUID_EXT, sizeof(uint64_t), &uuid, &size);

        if(uuid == query_uuid)
        {
            selected_device = devices[i];
            break;
        }
    }

    return selected_device;
}

static void create(const atcg::ref_ptr<Context>& shared, const int device_id, void** context_handle)
{
    EGLContext eglShared = EGL_NO_CONTEXT;

    if(shared)
    {
        eglShared = ((ContextData*)shared->getContextHandle())->context;
        shared->deactivate();
    }

    ATCG_ASSERT(!(*context_handle), "Handle already created");

    bool api = eglBindAPI(EGL_OPENGL_API);
    ATCG_ASSERT(api, "Could not bind opengl API");

    ContextData* data = new ContextData;
    data->display     = EGL_NO_DISPLAY;
    data->context     = EGL_NO_CONTEXT;
    data->surface     = EGL_NO_SURFACE;
    data->device_id   = device_id;
    *context_handle   = (void*)data;

    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglG"
                                                                                                                  "etPl"
                                                                                                                  "atfo"
                                                                                                                  "rmDi"
                                                                                                                  "spla"
                                                                                                                  "yEX"
                                                                                                                  "T");
    data->display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, detail::getDeviceFromIndex(device_id), NULL);
    ATCG_ASSERT(data->display != EGL_NO_DISPLAY, "Could not create EGL display");

    bool initialiazed = eglInitialize(data->display, nullptr, nullptr);
    ATCG_ASSERT(initialiazed, "Could not initialize EGL");

    // Choose EGL config
    EGLint configAttribs[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE};
    EGLConfig eglConfig;
    EGLint numConfigs;
    bool choose = eglChooseConfig(data->display, configAttribs, &eglConfig, 1, &numConfigs);
    ATCG_ASSERT(choose, "Failed to choose display");

    // Create an EGL context
    EGLint contextAttribs[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 6, EGL_NONE};
    data->context           = eglCreateContext(data->display, eglConfig, eglShared, contextAttribs);
    ATCG_ASSERT(data->context != EGL_NO_CONTEXT, "Failed to create context");
}
}    // namespace detail

void Context::destroy()
{
    if(_context_handle)
    {
        ContextData* data = (ContextData*)_context_handle;
        deactivate();
        eglDestroyContext(data->display, data->context);
        eglDestroySurface(data->display, data->surface);
        delete data;
        _context_handle = nullptr;
    }
}

void Context::create(const int device_id)
{
    detail::create(nullptr, device_id, &_context_handle);
    makeCurrent();
}

void Context::create(const atcg::ref_ptr<Context>& shared)
{
    int device_id = 0;
    if(shared)
    {
        device_id = ((ContextData*)shared->getContextHandle())->device_id;
    }
    detail::create(shared, device_id, &_context_handle);
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
    ContextData* data = (ContextData*)_context_handle;
    eglSwapBuffers(data->display, data->surface);
}

void Context::makeCurrent()
{
    ContextData* data = (ContextData*)_context_handle;
    eglMakeCurrent(data->display, data->surface, data->surface, data->context);
}

void Context::deactivate()
{
    ContextData* data = (ContextData*)_context_handle;
    eglMakeCurrent(data->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
}

bool Context::isCurrent() const
{
    ContextData* data = (ContextData*)_context_handle;
    return eglGetCurrentContext() == data->context;
}
}    // namespace atcg