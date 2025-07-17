#pragma once

#include <Asset/Asset.h>
#include <Renderer/Material.h>

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

    bool displayMaterial(const std::string& key, const atcg::ref_ptr<Material>& material);

    void displayGraph(AssetHandle handle);

    void displayScript(AssetHandle handle);

    void displayShader(AssetHandle handle);

    void displayTexture2D(AssetHandle handle);

private:
    AssetHandle _selected_handle = 0;

    std::string _current_vertex_path   = "";
    std::string _current_fragment_path = "";
    std::string _current_geometry_path = "";
    std::string _current_compute_path  = "";
};
}    // namespace GUI
}    // namespace atcg