#include <Asset/AssetPanel.h>

#ifndef ATCG_HEADLESS
    #include <implot.h>
#endif

#include <Core/Application.h>
#include <Core/glm.h>
#include <Asset/AssetManagerSystem.h>
#include <portable-file-dialogs.h>
#include <Scene/ComponentGUIHandler.h>
#include <Asset/Project.h>
#include <Scene/Scene.h>
#include <Core/Path.h>

namespace atcg
{
namespace GUI
{

AssetPanel::AssetPanel() {}

void AssetPanel::displayMaterial(AssetHandle handle) {}

void AssetPanel::displayGraph(AssetHandle handle) {}

void AssetPanel::displayScript(AssetHandle handle) {}

void AssetPanel::displayShader(AssetHandle handle) {}

void AssetPanel::displayTexture2D(AssetHandle handle) {}

void AssetPanel::displayTexture3D(AssetHandle handle) {}

void AssetPanel::displayScene(AssetHandle handle) {}


void AssetPanel::drawAssetList() {}

void AssetPanel::drawAdd() {}

void AssetPanel::drawAssetPanel() {}

void AssetPanel::drawAssetEditor() {}

void AssetPanel::renderPanel() {}

void AssetPanel::selectAsset(AssetHandle handle) {}

}    // namespace GUI
}    // namespace atcg