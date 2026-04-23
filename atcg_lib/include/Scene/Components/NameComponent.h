#pragma once

#include <string>

namespace atcg
{
struct NameComponent
{
    NameComponent() = default;
    NameComponent(const std::string& name) : _name(name) {}

    ATCG_INLINE const std::string& name() const { return _name; }

private:
    std::string _name;
};
}    // namespace atcg