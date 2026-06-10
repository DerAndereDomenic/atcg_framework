#pragma once

#include <string>
#include <Scene/ComponentSerializer.h>

namespace atcg
{
struct ATCG_API NameComponent
{
    NameComponent() = default;
    NameComponent(const std::string& name) : _name(name) {}

    ATCG_INLINE const std::string& name() const { return _name; }

private:
    std::string _name;
};

namespace Serialization
{
ATCG_DECLARE_COMPONENT_SERIALIZER(NameComponent);
}

}    // namespace atcg