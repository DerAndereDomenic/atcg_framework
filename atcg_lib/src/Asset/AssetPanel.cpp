#include <Asset/AssetPanel.h>

#ifndef ATCG_HEADLESS
    #include <implot.h>
#endif

#include <Asset/AssetManagerSystem.h>
#include <Scene/ComponentGUIHandler.h>
#include <portable-file-dialogs.h>

namespace atcg
{
namespace GUI
{

ATCG_INLINE void AssetPanel::drawAssetList()
{
    // Begin a scrollable horizontal region
    ImGui::BeginChild("AssetListHorizontal", ImVec2(0, 80), false, ImGuiWindowFlags_HorizontalScrollbar);

    const auto& registry = AssetManager::getAssetRegistry();

    for(auto entry: registry)
    {
        const auto& data = entry.second;
        auto handle      = entry.first;
        std::string tag  = data.name;

        if(handle == _selected_handle)
        {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.9f, 1.0f));    // Highlight color
        }

        ImGui::PushID(handle);
        if(ImGui::Button(tag.c_str()))
        {
            selectAsset(handle);
        }
        else if(handle == _selected_handle)
        {
            ImGui::PopStyleColor();
        }
        ImGui::PopID();


        ImGui::SameLine();    // Place next item on the same row
    }

    ImGui::EndChild();
}

void AssetPanel::renderPanel()
{
#ifndef ATCG_HEADLESS

    ImGui::Begin("Assets");

    if(ImGui::IsMouseDown(0) && ImGui::IsWindowHovered() && !ImGui::IsAnyItemHovered() && !ImGui::IsAnyItemActive())
    {
        selectAsset(0);
    }

    drawAssetList();

    ImGui::End();

    ImGui::Begin("Asset Editor");

    if(_selected_handle != 0)
    {
        const auto& data = AssetManager::getMetaData(_selected_handle);

        if(data.type == AssetType::Material)
        {
            displayMaterial("asset", atcg::AssetManager::getAsset<Material>(_selected_handle));
        }
        else if(data.type == AssetType::Graph)
        {
            if(ImGui::Button("Import Mesh##GeometryComponent"))
            {
                auto f     = pfd::open_file("Choose files to read",
                                        pfd::path::home(),
                                            {"Obj Files (.obj)", "*.obj"},
                                        pfd::opt::none);
                auto files = f.result();
                if(!files.empty())
                {
                    // RevisionStack::startRecording<ComponentEditedRevision<GeometryComponent>>(scene, entity);
                    auto mesh        = IO::read_any(files[0]);
                    auto graph_asset = AssetManager::getAsset<Graph>(_selected_handle);
                    graph_asset->copy(mesh);
                    // component.setGraph(mesh);
                    // atcg::RevisionStack::endRecording();
                }
            }
        }
    }

    ImGui::End();

#endif
}

void AssetPanel::selectAsset(AssetHandle handle)
{
    _selected_handle = handle;
}

}    // namespace GUI
}    // namespace atcg