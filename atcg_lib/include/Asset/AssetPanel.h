#pragma once

#include <Asset/Asset.h>

namespace atcg
{
namespace GUI
{
class AssetPanel
{
public:
    void renderPanel();

    void selectAsset(AssetHandle handle);

private:
    void drawAssetList();

    void drawAdd();

    AssetHandle _selected_handle = 0;
};
}    // namespace GUI
}    // namespace atcg