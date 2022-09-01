#include <Core/Window.h>

#include <glfw/glfw3.h>

#include <iostream>

#include <Events/KeyEvent.h>
#include <Events/WindowEvent.h>
#include <Events/MouseEvent.h>

namespace atcg
{
    static bool s_glfw_initialized = false;

    static void GLFWErrorCallback(int error, const char* description)
    {
        std::cerr << "GLFW Error: " << error << ": " << description << std::endl;
    }

    Window::Window(const WindowProps& props)
    {
        if(!s_glfw_initialized)
        {
            int success = glfwInit();
            if(success != GLFW_TRUE)
                return;
            
            s_glfw_initialized = true;
            glfwSetErrorCallback(GLFWErrorCallback);
        }

        _data.width = props.width;
        _data.height = props.height;

        _window = glfwCreateWindow((int)props.width, (int)props.height, props.title.c_str(), nullptr, nullptr);

        glfwSetWindowUserPointer(_window, &_data);

        // Set GLFW callbacks
		glfwSetWindowSizeCallback(_window, [](GLFWwindow* window, int width, int height)
		{
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
			data.width = width;
			data.height = height;

			WindowResizeEvent event(width, height);
			data.on_event(event);
		});

		glfwSetWindowCloseCallback(_window, [](GLFWwindow* window)
		{
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
			WindowCloseEvent event;
			data.on_event(event);
		});

		glfwSetKeyCallback(_window, [](GLFWwindow* window, int key, int scancode, int action, int mods)
		{
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

			switch (action)
			{
				case GLFW_PRESS:
				{
					KeyPressedEvent event(key, 0);
					data.on_event(event);
					break;
				}
				case GLFW_RELEASE:
				{
					KeyReleasedEvent event(key);
					data.on_event(event);
					break;
				}
				case GLFW_REPEAT:
				{
					KeyPressedEvent event(key, true);
					data.on_event(event);
					break;
				}
			}
		});

		glfwSetCharCallback(_window, [](GLFWwindow* window, unsigned int keycode)
		{
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

			KeyTypedEvent event(keycode);
			data.on_event(event);
		});

		glfwSetMouseButtonCallback(_window, [](GLFWwindow* window, int button, int action, int mods)
		{
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

			switch (action)
			{
				case GLFW_PRESS:
				{
					MouseButtonPressedEvent event(button);
					data.on_event(event);
					break;
				}
				case GLFW_RELEASE:
				{
					MouseButtonReleasedEvent event(button);
					data.on_event(event);
					break;
				}
			}
		});

		glfwSetScrollCallback(_window, [](GLFWwindow* window, double xOffset, double yOffset)
		{
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

			MouseScrolledEvent event((float)xOffset, (float)yOffset);
			data.on_event(event);
		});

		glfwSetCursorPosCallback(_window, [](GLFWwindow* window, double xPos, double yPos)
		{
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

			MouseMovedEvent event((float)xPos, (float)yPos);
			data.on_event(event);
		});

        glfwMakeContextCurrent(_window);
    }

    Window::~Window()
    {
        glfwDestroyWindow(_window);
    }

    void Window::onUpdate()
    {
        glfwPollEvents();
        glfwSwapBuffers(_window);
    }

    void Window::setEventCallback(const EventCallbackFn& callback)
    {
        _data.on_event = callback;
    }

    GLFWwindow* Window::getNativeWindow() const
    {
        return _window;
    }
}