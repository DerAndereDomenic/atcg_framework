#include <Core/Window.h>

#include <GLFW/glfw3.h>

#include <Renderer/Context.h>

namespace atcg
{
Window::Window(const WindowProps& props)
{
    _context = atcg::make_ref<Context>();
    _context->initWindowingAPI();

    _data.width  = props.width;
    _data.height = props.height;

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    _window = (void*)glfwCreateWindow((int)props.width, (int)props.height, props.title.c_str(), nullptr, nullptr);

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

    glfwSetWindowPos((GLFWwindow*)_window, pos_x, pos_y);

    _context->initGraphicsAPI(_window);

    glfwSetWindowUserPointer((GLFWwindow*)_window, &_data);

    // Set GLFW callbacks
    glfwSetWindowSizeCallback((GLFWwindow*)_window,
                              [](GLFWwindow* window, int width, int height)
                              {
                                  WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
                                  data.width       = std::max(width, 1);
                                  data.height      = std::max(height, 1);

                                  WindowResizeEvent event(data.width, data.height);
                                  data.on_event(&event);
                              });

    glfwSetWindowCloseCallback((GLFWwindow*)_window,
                               [](GLFWwindow* window)
                               {
                                   WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
                                   WindowCloseEvent event;
                                   data.on_event(&event);
                               });

    glfwSetKeyCallback((GLFWwindow*)_window,
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

    glfwSetCharCallback((GLFWwindow*)_window,
                        [](GLFWwindow* window, unsigned int keycode)
                        {
                            WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

                            KeyTypedEvent event(keycode);
                            data.on_event(&event);
                        });

    glfwSetMouseButtonCallback(
        (GLFWwindow*)_window,
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

    glfwSetScrollCallback((GLFWwindow*)_window,
                          [](GLFWwindow* window, double xOffset, double yOffset)
                          {
                              WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

                              MouseScrolledEvent event((float)xOffset, (float)yOffset);
                              data.on_event(&event);
                          });

    glfwSetCursorPosCallback((GLFWwindow*)_window,
                             [](GLFWwindow* window, double xPos, double yPos)
                             {
                                 WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

                                 data.current_mouse_x = (float)xPos;
                                 data.current_mouse_y = (float)yPos;

                                 MouseMovedEvent event((float)xPos, (float)yPos);
                                 data.on_event(&event);
                             });

    glfwSetDropCallback((GLFWwindow*)_window,
                        [](GLFWwindow* window, int path_count, const char* paths[])
                        {
                            WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

                            FileDroppedEvent event(paths[0]);
                            data.on_event(&event);
                        });

    toggleVSync(props.vsync);
    glfwShowWindow((GLFWwindow*)_window);
}

Window::~Window()
{
    glfwDestroyWindow((GLFWwindow*)_window);
    _context->deinitWindowingAPI();
}

void Window::onUpdate()
{
    glfwPollEvents();
    _context->swapBuffers();
}

void Window::setEventCallback(const EventCallbackFn& callback)
{
    _data.on_event = callback;
}

void* Window::getNativeWindow() const
{
    return _window;
}

void Window::resize(const uint32_t& _width, const uint32_t& _height)
{
    _data.width  = _width;
    _data.height = _height;

    glfwSetWindowSize((GLFWwindow*)_window, _width, _height);
}

void Window::toggleVSync(bool vsync)
{
    glfwSwapInterval(vsync);
}

glm::vec2 Window::getPosition() const
{
    int x;
    int y;
    glfwGetWindowPos((GLFWwindow*)_window, &x, &y);
    return glm::vec2(x, y);
}

float Window::getContentScale() const
{
    float xscale;
    glfwGetWindowContentScale((GLFWwindow*)_window, &xscale, NULL);
    return xscale;
}

void Window::hide()
{
    glfwHideWindow((GLFWwindow*)_window);
}

void Window::show()
{
    glfwShowWindow((GLFWwindow*)_window);
}
}    // namespace atcg