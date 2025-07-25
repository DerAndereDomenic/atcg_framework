#pragma once

#include <Asset/Asset.h>
#include <Renderer/Material.h>

namespace atcg
{
namespace GUI
{
/**
 * @brief Class to model an asset panel
 */
class AssetPanel
{
public:
    /**
     * @brief Render the panel
     */
    void renderPanel();

    /**
     * @brief Select an asset
     *
     * @param handle The handle
     */
    void selectAsset(AssetHandle handle);

private:
    void drawAssetPanel();

    void drawAssetEditor();

    void drawAssetList();

    void drawAdd();

    void displayMaterial(AssetHandle handle);

    void displayGraph(AssetHandle handle);

    void displayScript(AssetHandle handle);

    void displayShader(AssetHandle handle);

    void displayTexture2D(AssetHandle handle);

    void displayScene(AssetHandle handle);

private:
    AssetHandle _selected_handle = 0;

    std::string _current_vertex_path   = "";
    std::string _current_fragment_path = "";
    std::string _current_geometry_path = "";
    std::string _current_compute_path  = "";

    AssetType _panel_state = AssetType::None;
};
}    // namespace GUI
}    // namespace atcg