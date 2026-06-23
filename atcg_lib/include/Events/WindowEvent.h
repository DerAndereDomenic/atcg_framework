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
    FileDroppedEvent(const char** _paths, int _count) : _paths(_paths), _count(_count) {}

    std::string toString() const override
    {
        std::stringstream ss;
        ss << "FileDroppedEvent: " << _paths[0];
        return ss.str();
    }

    const char* getPath(int i = 0) const { return _paths[i]; }

    const char** getPaths() const { return _paths; }

    int getCount() const { return _count; }

    EVENT_CLASS_TYPE(FileDropped)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
private:
    const char** _paths;
    int _count;
};
}    // namespace atcg