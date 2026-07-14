#pragma once

#include <Plugin/PluginHandle.h>
#include <vector>

namespace atcg
{
template<typename Desc>
class Registry
{
public:
    using Handle = PluginHandle;

    bool registerType(Handle plugin, std::string_view type, Desc desc)
    {
        if(_entries.find(std::string(type)) != _entries.end()) return false;

        _entries.emplace(std::string(type), Entry {plugin, std::move(desc)});
        _registered_types.push_back(std::string(type));

        return true;
    }

    bool registerType(std::string_view type, Desc desc) { return registerType(nullptr, type, std::move(desc)); }

    void unregisterPlugin(Handle plugin)
    {
        // Erase matching entries from the map.
        for(auto it = _entries.begin(); it != _entries.end();)
        {
            if(it->second.plugin == plugin)
                it = _entries.erase(it);
            else
                ++it;
        }

        // Erase types that no longer exist in the map.
        _registered_types.erase(std::remove_if(_registered_types.begin(),
                                               _registered_types.end(),
                                               [&](const std::string& type)
                                               { return _entries.find(type) == _entries.end(); }),
                                _registered_types.end());
    }

    const Desc* find(std::string_view type) const
    {
        auto it = _entries.find(std::string(type));

        if(it == _entries.end()) return nullptr;

        return &it->second.desc;
    }

    const std::vector<std::string>& getRegisteredTypes() const { return _registered_types; }

private:
    struct Entry
    {
        Handle plugin;
        Desc desc;
    };

    std::unordered_map<std::string, Entry> _entries;
    std::vector<std::string> _registered_types;
};
}    // namespace atcg