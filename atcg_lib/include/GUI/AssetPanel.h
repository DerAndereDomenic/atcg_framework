#pragma once

#include <Core/API.h>
#include <Asset/Asset.h>
#include <Renderer/Material.h>
#include <Renderer/Framebuffer.h>
#include <Scene/Scene.h>
#include <Scene/Entity.h>

namespace atcg
{
namespace GUI
{
/**
 * @brief Class to model an asset panel
 */
class ATCG_API AssetPanel
{
public:
    /**
     * @brief Constructor
     */
    AssetPanel();

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

    /**
     * @brief Get the selected asset, i.e., the asset that is open in the asset editor.
     *
     * @return The handle of the selected asset or 0 if no asset is selected
     */
    ATCG_INLINE AssetHandle getSelectedAsset() const { return _selected_handle; }

    /**
     * @brief Check if the selected asset is the most recent selection.
     * The asset stays selected when entities of the active scene are selected afterwards, but it is no longer the most
     * recent selection. Like this, the application can decide whether an action (e.g. copy) refers to the selected
     * asset or to the selected entities.
     *
     * @return True if an asset is selected and no entities were selected after it
     */
    ATCG_INLINE bool isAssetSelectionActive() const { return _selected_handle != 0 && _asset_selection_active; }

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

    void displayTexture3D(AssetHandle handle);

    void displayScene(AssetHandle handle);

private:
    AssetHandle _selected_handle = 0;

    // Whether the asset or the entities were selected most recently
    bool _asset_selection_active = false;
    std::vector<Entity> _last_entity_selection;

    std::string _current_vertex_path   = "";
    std::string _current_fragment_path = "";
    std::string _current_geometry_path = "";
    std::string _current_compute_path  = "";

    TextureSpecification _spec_3d;
    std::string _current_texture_3d_path = "";
    uint32_t _slice                      = 0;
    atcg::ref_ptr<Texture2D> _preview;

    AssetType _panel_state = AssetType::None;

    atcg::ref_ptr<Texture2D> _folder_icon;
    atcg::ref_ptr<Texture2D> _script_icon;
    atcg::ref_ptr<Texture2D> _material_icon;
    atcg::ref_ptr<Texture2D> _mesh_icon;
    atcg::ref_ptr<Texture2D> _image_icon;

    atcg::ref_ptr<Scene> _preview_scene;

    atcg::ref_ptr<Framebuffer> _preview_framebuffer;
    atcg::ref_ptr<Material> _preview_material;
};
}    // namespace GUI
}    // namespace atcg