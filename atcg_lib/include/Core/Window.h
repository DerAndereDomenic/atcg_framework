#pragma once

#include <sstream>
#include <functional>

#include <Events/Event.h>

struct GLFWwindow;

namespace atcg
{
struct WindowProps
{
    std::string title;
    uint32_t width;
    uint32_t height;

    WindowProps(const std::string& title = "ATCG", uint32_t width = 1600, uint32_t height = 900)
        : title(title),
          width(width),
          height(height)
    {
    }
};

/**
 * @brief Class to model a window
 *
 */
class Window
{
public:
    // A function pointer that is used on event
    using EventCallbackFn = std::function<void(Event&)>;

    /**
     * @brief Construct a new Window object
     *
     * @param props The window properties
     */
    Window(const WindowProps& props);

    /**
     * @brief Destroy the Window object
     *
     */
    ~Window();

    /**
     * @brief Handles the swap chain and other per frame properties of the renderer/window
     *
     */
    void onUpdate();

    /**
     * @brief Set the Event Callback object
     *
     * @param callback The callback function
     */
    void setEventCallback(const EventCallbackFn& callback);

    /**
     * @brief Get the Native Window object
     *
     * @return GLFWwindow* The glfw window pointer
     */
    GLFWwindow* getNativeWindow() const;

    /**
     * @brief Get the Width object
     *
     * @return uint32_t width
     */
    inline uint32_t getWidth() const { return _data.width; }

    /**
     * @brief Get the Height object
     *
     * @return uint32_t height
     */
    inline uint32_t getHeight() const { return _data.height; }

private:
    struct WindowData
    {
        uint32_t width;
        uint32_t height;
        EventCallbackFn on_event;
    };

    WindowData _data;
    GLFWwindow* _window;
};
}    // namespace atcg