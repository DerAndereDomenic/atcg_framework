#pragma once

#include <Events/Event.h>
#include <sstream>

// Based on Hazel Engine (https://github.com/TheCherno/Hazel)
// Modified by Domenic Zingsheim in 2023

namespace atcg
{

class WindowResizeEvent : public Event
{
public:
    WindowResizeEvent(unsigned int width, unsigned int height) : _width(width), _height(height) {}

    unsigned int getWidth() const { return _width; }
    unsigned int getHeight() const { return _height; }

    std::string toString() const override
    {
        std::stringstream ss;
        ss << "WindowResizeEvent: " << _width << ", " << _height;
        return ss.str();
    }

    EVENT_CLASS_TYPE(WindowResize)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
private:
    unsigned int _width, _height;
};

class ViewportResizeEvent : public Event
{
public:
    ViewportResizeEvent(unsigned int width, unsigned int height) : _width(width), _height(height) {}

    unsigned int getWidth() const { return _width; }
    unsigned int getHeight() const { return _height; }

    std::string toString() const override
    {
        std::stringstream ss;
        ss << "ViewportResizeEvent: " << _width << ", " << _height;
        return ss.str();
    }

    EVENT_CLASS_TYPE(ViewportResize)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
private:
    unsigned int _width, _height;
};

class WindowCloseEvent : public Event
{
public:
    WindowCloseEvent() = default;

    EVENT_CLASS_TYPE(WindowClose)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
};

class FileDroppedEvent : public Event
{
public:
    FileDroppedEvent(const std::string& path) : _path(path) {}

    std::string toString() const override
    {
        std::stringstream ss;
        ss << "FileDroppedEvent: " << _path;
        return ss.str();
    }

    const std::string& getPath() const { return _path; }

    EVENT_CLASS_TYPE(FileDropped)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
private:
    std::string _path;
};
}    // namespace atcg