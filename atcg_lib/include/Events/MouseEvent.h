#pragma once

#include <Events/Event.h>
#include <sstream>

// Based on Hazel Engine (https://github.com/TheCherno/Hazel)
// Modified by Domenic Zingsheim in 2023

namespace atcg
{

class MouseMovedEvent : public Event
{
public:
    MouseMovedEvent(const float x, const float y) : _mouseX(x), _mouseY(y) {}

    float getX() const { return _mouseX; }
    float getY() const { return _mouseY; }

    std::string toString() const override
    {
        std::stringstream ss;
        ss << "MouseMovedEvent: " << _mouseX << ", " << _mouseY;
        return ss.str();
    }

    EVENT_CLASS_TYPE(MouseMoved)
    EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)
private:
    float _mouseX, _mouseY;
};

class MouseScrolledEvent : public Event
{
public:
    MouseScrolledEvent(const float xOffset, const float yOffset) : _offsetX(xOffset), _offsetY(yOffset) {}

    float getXOffset() const { return _offsetX; }
    float getYOffset() const { return _offsetY; }

    std::string toString() const override
    {
        std::stringstream ss;
        ss << "MouseScrolledEvent: " << getXOffset() << ", " << getYOffset();
        return ss.str();
    }

    EVENT_CLASS_TYPE(MouseScrolled)
    EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)
private:
    float _offsetX, _offsetY;
};

class MouseButtonEvent : public Event
{
public:
    int32_t getMouseButton() const { return _button; }

    float getX() const { return _mouseX; }
    float getY() const { return _mouseY; }

    EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput | EventCategoryMouseButton)
protected:
    MouseButtonEvent(const int32_t button, const float x, const float y) : _button(button), _mouseX(x), _mouseY(y) {}

    int32_t _button;
    float _mouseX;
    float _mouseY;
};

class MouseButtonPressedEvent : public MouseButtonEvent
{
public:
    MouseButtonPressedEvent(const int32_t button, const float x, const float y) : MouseButtonEvent(button, x, y) {}

    std::string toString() const override
    {
        std::stringstream ss;
        ss << "MouseButtonPressedEvent: " << _button;
        return ss.str();
    }

    EVENT_CLASS_TYPE(MouseButtonPressed)
};

class MouseButtonReleasedEvent : public MouseButtonEvent
{
public:
    MouseButtonReleasedEvent(const int32_t button, const float x, const float y) : MouseButtonEvent(button, x, y) {}

    std::string toString() const override
    {
        std::stringstream ss;
        ss << "MouseButtonReleasedEvent: " << _button;
        return ss.str();
    }

    EVENT_CLASS_TYPE(MouseButtonReleased)
};

}    // namespace atcg