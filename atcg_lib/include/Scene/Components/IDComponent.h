#pragma once

#include <Core/UUID.h>
#include <Scene/ComponentSerializer.h>

namespace atcg
{
struct IDComponent
{
    IDComponent() : _ID(UUID()) {}
    IDComponent(uint64_t id) : _ID(UUID(id)) {}

    ATCG_INLINE UUID ID() const { return _ID; }

private:
    UUID _ID;
};

namespace Serialization
{
ATCG_DECLARE_COMPONENT_SERIALIZER(IDComponent);
}

}    // namespace atcg