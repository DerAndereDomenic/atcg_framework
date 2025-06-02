#pragma once

#include <Events/Event.h>
#include <sstream>

// Based on Hazel Engine (https://github.com/TheCherno/Hazel)
// Modified by Domenic Zingsheim in 2023

namespace atcg
{
class VRButtonEvent : public Event
{
public:
    uint32_t getVRButton() const { return _button; }
    uint32_t getDeviceIndex() const { return _device_index; }

    EVENT_CLASS_CATEGORY(EventCategoryVR | EventCategoryInput | EventCategroyVRButton)
protected:
    VRButtonEvent(const uint32_t button, const uint32_t device) : _device_index(device), _button(button) {}

    uint32_t _device_index;
    uint32_t _button;
};

class VRButtonPressedEvent : public VRButtonEvent
{
public:
    VRButtonPressedEvent(const uint32_t button, const uint32_t device) : VRButtonEvent(button, device) {}

    std::string toString() const override
    {
        std::stringstream ss;
        ss << "VRButtonPressedEvent: " << _button << "; Device: " << _device_index;
        return ss.str();
    }

    EVENT_CLASS_TYPE(VRButtonPressed)
};

class VRButtonReleasedEvent : public VRButtonEvent
{
public:
    VRButtonReleasedEvent(const uint32_t button, const uint32_t device) : VRButtonEvent(button, device) {}

    std::string toString() const override
    {
        std::stringstream ss;
        ss << "VRButtonReleasedEvent: " << _button << "; Device: " << _device_index;
        return ss.str();
    }

    EVENT_CLASS_TYPE(VRButtonReleased)
};

class VRButtonTouchedEvent : public VRButtonEvent
{
public:
    VRButtonTouchedEvent(const uint32_t button, const uint32_t device) : VRButtonEvent(button, device) {}

    std::string toString() const override
    {
        std::stringstream ss;
        ss << "VRButtonTouchedEvent: " << _button << "; Device: " << _device_index;
        return ss.str();
    }

    EVENT_CLASS_TYPE(VRButtonTouched)
};

class VRButtonUntouchedEvent : public VRButtonEvent
{
public:
    VRButtonUntouchedEvent(const uint32_t button, const uint32_t device) : VRButtonEvent(button, device) {}

    std::string toString() const override
    {
        std::stringstream ss;
        ss << "VRButtonUntouchedEvent: " << _button << "; Device: " << _device_index;
        return ss.str();
    }

    EVENT_CLASS_TYPE(VRButtonUntouched)
};
}    // namespace atcg