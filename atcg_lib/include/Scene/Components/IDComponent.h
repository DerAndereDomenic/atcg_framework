#pragma once

#include <Core/UUID.h>
#include <Scene/ComponentSerializer.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{
struct ATCG_API IDComponent
{
    IDComponent() : _ID(UUID()) {}
    IDComponent(uint64_t id) : _ID(UUID(id)) {}

    ATCG_INLINE UUID ID() const { return _ID; }

    static void registerComponent(ComponentRegistry::Registry* registry);

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "ID"; }

private:
    UUID _ID;
};

namespace GUI
{
template<>
struct is_gui_addable<IDComponent> : std::false_type
{
};

template<>
struct is_gui_renderable<IDComponent> : std::false_type
{
};
}    // namespace GUI

namespace Serialization
{
ATCG_DECLARE_COMPONENT_SERIALIZER(IDComponent);
}

}    // namespace atcg