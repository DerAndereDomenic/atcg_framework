#pragma once

#include <Events/Event.h>
#include <sstream>

namespace atcg {

	class KeyEvent : public Event
	{
	public:
		int32_t getKeyCode() const { return _keycode; }

		EVENT_CLASS_CATEGORY(EventCategoryKeyboard | EventCategoryInput)
	protected:
		KeyEvent(const int32_t keycode)
			: _keycode(keycode) {}

		int32_t _keycode;
	};

	class KeyPressedEvent : public KeyEvent
	{
	public:
		KeyPressedEvent(const int32_t keycode, bool isRepeat = false)
			: KeyEvent(keycode), _is_repeat(isRepeat) {}

		bool IsRepeat() const { return _is_repeat; }

		std::string toString() const override
		{
			std::stringstream ss;
			ss << "KeyPressedEvent: " << _keycode << " (repeat = " << _is_repeat << ")";
			return ss.str();
		}

		EVENT_CLASS_TYPE(KeyPressed)
	private:
		bool _is_repeat;
	};

	class KeyReleasedEvent : public KeyEvent
	{
	public:
		KeyReleasedEvent(const int32_t int32_t)
			: KeyEvent(int32_t) {}

		std::string toString() const override
		{
			std::stringstream ss;
			ss << "KeyReleasedEvent: " << _keycode;
			return ss.str();
		}

		EVENT_CLASS_TYPE(KeyReleased)
	};

	class KeyTypedEvent : public KeyEvent
	{
	public:
		KeyTypedEvent(const int32_t int32_t)
			: KeyEvent(int32_t) {}

		std::string toString() const override
		{
			std::stringstream ss;
			ss << "KeyTypedEvent: " << _keycode;
			return ss.str();
		}

		EVENT_CLASS_TYPE(KeyTyped)
	};
}