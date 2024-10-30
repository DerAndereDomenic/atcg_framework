#pragma once


#include <Core/glm.h>
#include <Core/Memory.h>
#include <Events/Event.h>
#include <Renderer/Context.h>

#include <sstream>
#include <functional>
#include <numeric>

namespace atcg
{

struct WindowProps
{
    std::string title;
    uint32_t width;
    uint32_t height;
    int32_t pos_x;
    int32_t pos_y;
    bool vsync;
    bool hidden;

    WindowProps(const std::string& title = "ATCG",
                uint32_t width           = 1600,
                uint32_t height          = 900,
                int32_t pos_x            = std::numeric_limits<int32_t>::max(),
                int32_t pos_y            = std::numeric_limits<int32_t>::max(),
                bool vsync               = true,
                bool hidden              = false)
        : title(title),
          width(width),
          height(height),
          pos_x(pos_x),
          pos_y(pos_y),
          vsync(vsync),
          hidden(hidden)
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
    using EventCallbackFn = std::function<void(Event*)>;

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
     * @brief Hides the window.
     *
     */
    void hide();

    /**
     * @brief Shows the window
     *
     */
    void show();

    /**
     * @brief Get the Native Window object
     *
     * @return GLFWwindow* The glfw window pointer
     */
    void* getNativeWindow() const;

    /**
     * @brief Resize the window
     *
     * @param width The width
     * @param height The height
     */
    void resize(const uint32_t& width, const uint32_t& height);

    /**
     * @brief Toggle if vsync should be enabled or not
     *
     * @param vsyinc if vsync should be enabled
     */
    void toggleVSync(bool vsync);

    /**
     * @brief Toggle fullscreen mode of the window.
     * If the window is already in fullscreen, it will go back to window mode.
     */
    void toggleFullscreen();

    /**
     * @brief Get the window position
     *
     * @return Vector of (x,y) with the absolute window coordinates
     */
    glm::vec2 getPosition() const;

    /**
     * @brief Get the content scale of the window (4k support)
     *
     * @return The content scaling factor
     */
    float getContentScale() const;

    /**
     * @brief Get the Width object
     *
     * @return uint32_t width
     */
    ATCG_INLINE uint32_t getWidth() const { return _data.width; }

    /**
     * @brief Get the Height object
     *
     * @return uint32_t height
     */
    ATCG_INLINE uint32_t getHeight() const { return _data.height; }

    /**
     * @brief Check if the window is currently in fullscreen mode
     *
     * @return True if in fullscreen
     */
    ATCG_INLINE bool isFullScreen() const { return _data.fullscreen; }

    /**
     * @brief Get the context
     *
     * @return The context
     */
    ATCG_INLINE atcg::ref_ptr<Context> getContext() const { return _context; }

private:
    struct WindowData
    {
        uint32_t width;
        uint32_t height;
        float current_mouse_x;
        float current_mouse_y;
        EventCallbackFn on_event;
        bool fullscreen;
        uint32_t fullscreen_width;
        uint32_t fullscreen_height;
        int32_t fullscreen_x;
        int32_t fullscreen_y;
    };

    WindowData _data;
    atcg::ref_ptr<Context> _context;
};
}    // namespace atcg