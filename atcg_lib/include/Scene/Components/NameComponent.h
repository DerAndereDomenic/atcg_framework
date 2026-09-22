#pragma once

#include <string>
#include <Scene/ComponentSerializer.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{
struct ATCG_API NameComponent
{
    NameComponent() = default;
    NameComponent(const std::string& name) : _name(name) {}

    ATCG_INLINE const std::string& name() const { return _name; }

    static void registerComponent(ComponentRegistry::Registry* registry);

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Name"; }

private:
    std::string _name;
};

namespace GUI
{
template<>
struct is_gui_addable<NameComponent> : std::false_type
{
};

template<>
struct is_gui_renderable<NameComponent> : std::false_type
{
};
}    // namespace GUI

namespace Serialization
{
ATCG_DECLARE_COMPONENT_SERIALIZER(NameComponent);
}

}    // namespace atcg