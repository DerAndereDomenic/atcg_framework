#pragma once

#include <string>
#include <ostream>
#include <Core/Platform.h>

// Based on Hazel Engine (https://github.com/TheCherno/Hazel)
// Modified by Domenic Zingsheim in 2023

namespace atcg
{
enum class EventType
{
    None = 0,
    WindowClose,
    WindowResize,
    WindowFocus,
    WindowLostFocus,
    WindowMoved,
    ViewportResize,
    KeyPressed,
    KeyReleased,
    KeyTyped,
    MouseButtonPressed,
    MouseButtonReleased,
    MouseMoved,
    MouseScrolled,
    FileDropped,
    VRButtonPressed,
    VRButtonReleased,
    VRButtonTouched,
    VRButtonUntouched
};

#define BIT(x) (1 << x)
enum EventCategory
{
    None                     = 0,
    EventCategoryApplication = BIT(0),
    EventCategoryInput       = BIT(1),
    EventCategoryKeyboard    = BIT(2),
    EventCategoryMouse       = BIT(3),
    EventCategoryMouseButton = BIT(4),
    EventCategoryVR          = BIT(5),
    EventCategroyVRButton    = BIT(6)
};
#undef BIT

#define EVENT_CLASS_TYPE(type)                                                                                         \
    static EventType getStaticType()                                                                                   \
    {                                                                                                                  \
        return EventType::type;                                                                                        \
    }                                                                                                                  \
    virtual EventType getEventType() const override                                                                    \
    {                                                                                                                  \
        return getStaticType();                                                                                        \
    }                                                                                                                  \
    virtual const char* getName() const override                                                                       \
    {                                                                                                                  \
        return #type;                                                                                                  \
    }

#define EVENT_CLASS_CATEGORY(category)                                                                                 \
    virtual int getCategoryFlags() const override                                                                      \
    {                                                                                                                  \
        return category;                                                                                               \
    }

class Event
{
public:
    virtual ~Event() = default;

    bool handled = false;

    virtual EventType getEventType() const = 0;
    virtual const char* getName() const    = 0;
    virtual int getCategoryFlags() const   = 0;
    virtual std::string toString() const { return getName(); }

    bool isInCategory(EventCategory category) { return getCategoryFlags() & category; }
};

class EventDispatcher
{
public:
    EventDispatcher(Event* event) : _event(event) {}

    // F will be deduced by the compiler
    template<typename T, typename F>
    bool dispatch(const F& func)
    {
        if(_event->getEventType() == T::getStaticType())
        {
            _event->handled |= func(static_cast<T*>(_event));
            return true;
        }
        return false;
    }

private:
    Event* _event;
};

ATCG_INLINE std::ostream& operator<<(std::ostream& os, const Event& e)
{
    return os << e.toString();
}
}    // namespace atcg