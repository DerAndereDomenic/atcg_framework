#pragma once

#include <Core/API.h>
#include <Core/Memory.h>

namespace atcg
{
namespace GUI
{
class ATCG_API PluginPanel
{
public:
    PluginPanel() = default;

    ~PluginPanel() = default;

    void renderPanel();

private:
};
}    // namespace GUI
}    // namespace atcg