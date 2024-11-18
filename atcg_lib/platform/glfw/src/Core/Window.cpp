#include <Core/Window.h>
#include <Core/Assert.h>

#include <GLFW/glfw3.h>
#include <Core/Path.h>

#include <Renderer/Context.h>

#include <stb_image.h>

namespace atcg
{
namespace detail
{
static bool s_glfw_initialized = false;

static void GLFWErrorCallback(int error, const char* description)
{
    ATCG_ERROR("GLFW Error: {0}: {1}", error, description);
}
}    // namespace detail

Window::Window(const WindowProps& props)
{
    _context = atcg::make_ref<Context>();

    // Initialize glfw
    if(!detail::s_glfw_initialized)
    {
        int success = glfwInit();
        if(success != GLFW_TRUE) return;

        detail::s_glfw_initialized = true;
        glfwSetErrorCallback(detail::GLFWErrorCallback);
    }

    _context->create();
    _context->initGraphicsAPI();

    _data.width      = props.width;
    _data.height     = props.height;
    _data.fullscreen = false;

    void* window = _context->getContextHandle();
    glfwSetWindowTitle((GLFWwindow*)window, props.title.c_str());
    glfwSetWindowSize((GLFWwindow*)window, props.width, props.height);

    int pos_x = props.pos_x;
    int pos_y = props.pos_y;
    if(pos_x == std::numeric_limits<int32_t>::max() || pos_y == std::numeric_limits<int32_t>::max())
    {
        int monitorX, monitorY;
        GLFWmonitor* monitor         = glfwGetPrimaryMonitor();
        const GLFWvidmode* videoMode = glfwGetVideoMode(monitor);
        glfwGetMonitorPos(monitor, &monitorX, &monitorY);

        pos_x = monitorX + (videoMode->width - _data.width) / 2;
        pos_y = monitorY + (videoMode->height - _data.height) / 2;
    }

    glfwSetWindowPos((GLFWwindow*)window, pos_x, pos_y);

    glfwSetWindowUserPointer((GLFWwindow*)window, &_data);

    // Set GLFW callbacks
    glfwSetWindowSizeCallback((GLFWwindow*)window,
                              [](GLFWwindow* window, int width, int height)
                              {
                                  WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
                                  data.width       = std::max(width, 1);
                                  data.height      = std::max(height, 1);

                                  WindowResizeEvent event(data.width, data.height);
                                  data.on_event(&event);
                              });

    glfwSetWindowCloseCallback((GLFWwindow*)window,
                               [](GLFWwindow* window)
                               {
                                   WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
                                   WindowCloseEvent event;
                                   data.on_event(&event);
                               });

    glfwSetKeyCallback((GLFWwindow*)window,
                       [](GLFWwindow* window, int key, int scancode, int action, int mods)
                       {
                           WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

                           switch(action)
                           {
                               case GLFW_PRESS:
                               {
                                   KeyPressedEvent event(key, 0);
                                   data.on_event(&event);
                                   break;
                               }
                               case GLFW_RELEASE:
                               {
                                   KeyReleasedEvent event(key);
                                   data.on_event(&event);
                                   break;
                               }
                               case GLFW_REPEAT:
                               {
                                   KeyPressedEvent event(key, true);
                                   data.on_event(&event);
                                   break;
                               }
                           }
                       });

    glfwSetCharCallback((GLFWwindow*)window,
                        [](GLFWwindow* window, unsigned int keycode)
                        {
                            WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

                            KeyTypedEvent event(keycode);
                            data.on_event(&event);
                        });

    glfwSetMouseButtonCallback(
        (GLFWwindow*)window,
        [](GLFWwindow* window, int button, int action, int mods)
        {
            WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

            switch(action)
            {
                case GLFW_PRESS:
                {
                    MouseButtonPressedEvent event(button, data.current_mouse_x, data.current_mouse_y);
                    data.on_event(&event);
                    break;
                }
                case GLFW_RELEASE:
                {
                    MouseButtonReleasedEvent event(button, data.current_mouse_x, data.current_mouse_y);
                    data.on_event(&event);
                    break;
                }
            }
        });

    glfwSetScrollCallback((GLFWwindow*)window,
                          [](GLFWwindow* window, double xOffset, double yOffset)
                          {
                              WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

                              MouseScrolledEvent event((float)xOffset, (float)yOffset);
                              data.on_event(&event);
                          });

    glfwSetCursorPosCallback((GLFWwindow*)window,
                             [](GLFWwindow* window, double xPos, double yPos)
                             {
                                 WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

                                 data.current_mouse_x = (float)xPos;
                                 data.current_mouse_y = (float)yPos;

                                 MouseMovedEvent event((float)xPos, (float)yPos);
                                 data.on_event(&event);
                             });

    glfwSetDropCallback((GLFWwindow*)window,
                        [](GLFWwindow* window, int path_count, const char* paths[])
                        {
                            WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

                            FileDroppedEvent event(paths[0]);
                            data.on_event(&event);
                        });

    toggleVSync(props.vsync);

    // Load Icon
    std::string icon_path = props.icon_path;
    if(props.icon_path == "")
    {
        icon_path = (atcg::resource_directory() / "icon.png").generic_string();
    }

    int width, height, channels;
    unsigned char* pixels = stbi_load(icon_path.c_str(), &width, &height, &channels, 4);    // Load as RGBA
    if(pixels)
    {
        GLFWimage icon;
        icon.width  = width;
        icon.height = height;
        icon.pixels = pixels;

        // Set the icon for the GLFW window
        glfwSetWindowIcon((GLFWwindow*)window, 1, &icon);

        // Free the loaded image data
        stbi_image_free(pixels);
    }
    else
    {
        // Handle error if the icon can't be loaded
        ATCG_ERROR("Failed to load icon image {}", icon_path);
    }

    if(!props.hidden) show();
}

Window::~Window()
{
    _context->destroy();
    if(detail::s_glfw_initialized)
    {
        glfwTerminate();
        detail::s_glfw_initialized = false;
    }
}

void Window::onUpdate()
{
    ATCG_ASSERT(_context, "No valid context");

    glfwPollEvents();
    _context->swapBuffers();
}

void Window::setEventCallback(const EventCallbackFn& callback)
{
    _data.on_event = callback;
}

void* Window::getNativeWindow() const
{
    return _context->getContextHandle();
}

void Window::resize(const uint32_t& _width, const uint32_t& _height)
{
    _data.width  = _width;
    _data.height = _height;

    glfwSetWindowSize((GLFWwindow*)_context->getContextHandle(), _width, _height);
}

void Window::toggleVSync(bool vsync)
{
    glfwSwapInterval(vsync);
}

void Window::toggleFullscreen()
{
    if(!_data.fullscreen)
    {
        // Store for exiting fullscreen
        _data.fullscreen_width  = _data.width;
        _data.fullscreen_height = _data.height;
        _data.fullscreen_x      = getPosition().x;
        _data.fullscreen_y      = getPosition().y;

        // Find best monitor
        int monitor_count;
        GLFWmonitor** monitors    = glfwGetMonitors(&monitor_count);
        GLFWmonitor* best_monitor = nullptr;

        int maxOverlapArea = 0;
        for(int i = 0; i < monitor_count; ++i)
        {
            int mx, my, mw, mh;
            glfwGetMonitorWorkarea(monitors[i], &mx, &my, &mw, &mh);

            int overlapWidth  = std::max(0,
                                        std::min((int)_data.fullscreen_x + (int)_data.fullscreen_width, mx + mw) -
                                            std::max(_data.fullscreen_x, mx));
            int overlapHeight = std::max(0,
                                         std::min((int)_data.fullscreen_y + (int)_data.fullscreen_height, my + mh) -
                                             std::max(_data.fullscreen_y, my));
            int overlapArea   = overlapWidth * overlapHeight;

            if(overlapArea > maxOverlapArea)
            {
                maxOverlapArea = overlapArea;
                best_monitor   = monitors[i];
            }
        }

        int x, y, width, height;
        glfwGetMonitorWorkarea(best_monitor, &x, &y, &width, &height);
        glfwSetWindowMonitor((GLFWwindow*)_context->getContextHandle(),
                             best_monitor,
                             x,
                             y,
                             width,
                             height,
                             GLFW_DONT_CARE);
        _data.fullscreen = true;
    }
    else
    {
        glfwSetWindowMonitor((GLFWwindow*)_context->getContextHandle(),
                             NULL,
                             _data.fullscreen_x,
                             _data.fullscreen_y,
                             _data.fullscreen_width,
                             _data.fullscreen_height,
                             GLFW_DONT_CARE);
        _data.fullscreen = false;
    }
}

glm::vec2 Window::getPosition() const
{
    int x;
    int y;
    glfwGetWindowPos((GLFWwindow*)_context->getContextHandle(), &x, &y);
    return glm::vec2(x, y);
}

void Window::setPosition(const glm::vec2& position)
{
    glfwSetWindowPos((GLFWwindow*)_context->getContextHandle(), (int)position.x, (int)position.y);
}

float Window::getContentScale() const
{
    float xscale;
    glfwGetWindowContentScale((GLFWwindow*)_context->getContextHandle(), &xscale, NULL);
    return xscale;
}

void Window::hide()
{
    glfwHideWindow((GLFWwindow*)_context->getContextHandle());
}

void Window::show()
{
    glfwShowWindow((GLFWwindow*)_context->getContextHandle());
}
}    // namespace atcg