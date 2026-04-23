#pragma once

#include <Core/UUID.h>

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
}    // namespace atcg