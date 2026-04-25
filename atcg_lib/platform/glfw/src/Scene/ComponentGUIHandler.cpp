#include <Scene/ComponentGUIHandler.h>

#include <Core/Application.h>
#include <Scene/Entity.h>
#include <Utils/Utils.h>

#include <imgui.h>
#include <portable-file-dialogs.h>

namespace atcg
{
namespace GUI
{

AssetHandle displayMaterialSelection(const std::string& key, AssetHandle handle)
{
    const auto& data = AssetManager::getMetaData(handle);

    std::string tag = AssetManager::isAssetHandleValid(handle) ? data.name : "Default Material";

    const auto& registry = AssetManager::getAssetRegistry();

    AssetHandle current_item = handle;

    if(ImGui::BeginCombo(("Select Material##" + key).c_str(), tag.c_str()))
    {
        // No Selection
        {
            bool is_selected = !AssetManager::isAssetHandleValid(current_item);

            if(ImGui::Selectable("Default Material", is_selected))
            {
                current_item = 0;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        for(auto it = registry.begin(); it != registry.end(); ++it)
        {
            if(it->second.type != AssetType::Material) continue;

            bool is_selected = it->first == current_item;

            if(ImGui::Selectable((it->second.name + "##" + std::to_string(it->first)).c_str(), is_selected))
            {
                current_item = it->first;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        ImGui::EndCombo();
    }

    return current_item;
}

AssetHandle displayGraphSelection(const std::string& key, AssetHandle handle)
{
    const auto& data = AssetManager::getMetaData(handle);

    std::string tag = AssetManager::isAssetHandleValid(handle) ? data.name : "No Graph";

    const auto& registry = AssetManager::getAssetRegistry();

    AssetHandle current_item = handle;

    if(ImGui::BeginCombo(("Select Graph##" + key).c_str(), tag.c_str()))
    {
        // No Selection
        {
            bool is_selected = !AssetManager::isAssetHandleValid(current_item);

            if(ImGui::Selectable("No Graph", is_selected))
            {
                current_item = 0;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        for(auto it = registry.begin(); it != registry.end(); ++it)
        {
            if(it->second.type != AssetType::Graph) continue;

            bool is_selected = it->first == current_item;

            if(ImGui::Selectable((it->second.name + "##" + std::to_string(it->first)).c_str(), is_selected))
            {
                current_item = it->first;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        ImGui::EndCombo();
    }

    return current_item;
}

AssetHandle displayScriptSelection(const std::string& key, AssetHandle handle)
{
    const auto& data = AssetManager::getMetaData(handle);

    std::string tag = AssetManager::isAssetHandleValid(handle) ? data.name : "No Script";

    const auto& registry = AssetManager::getAssetRegistry();

    AssetHandle current_item = handle;

    if(ImGui::BeginCombo(("Select Script##" + key).c_str(), tag.c_str()))
    {
        // No Selection
        {
            bool is_selected = !AssetManager::isAssetHandleValid(current_item);

            if(ImGui::Selectable("No Script", is_selected))
            {
                current_item = 0;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        for(auto it = registry.begin(); it != registry.end(); ++it)
        {
            if(it->second.type != AssetType::Script) continue;

            bool is_selected = it->first == current_item;

            if(ImGui::Selectable((it->second.name + "##" + std::to_string(it->first)).c_str(), is_selected))
            {
                current_item = it->first;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        ImGui::EndCombo();
    }

    return current_item;
}

AssetHandle displayShaderSelection(const std::string& key, AssetHandle handle)
{
    const auto& data = AssetManager::getMetaData(handle);

    std::string tag = AssetManager::isAssetHandleValid(handle) ? data.name : "Default Shader";

    const auto& registry = AssetManager::getAssetRegistry();

    AssetHandle current_item = handle;

    if(ImGui::BeginCombo(("Select Shader##" + key).c_str(), tag.c_str()))
    {
        // No Selection
        {
            bool is_selected = !AssetManager::isAssetHandleValid(current_item);

            if(ImGui::Selectable("Default Shader", is_selected))
            {
                current_item = 0;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        for(auto it = registry.begin(); it != registry.end(); ++it)
        {
            if(it->second.type != AssetType::Shader) continue;

            auto shader = AssetManager::getAsset<Shader>(it->first);
            if(shader && shader->isComputeShader()) continue;

            bool is_selected = it->first == current_item;

            if(ImGui::Selectable((it->second.name + "##" + std::to_string(it->first)).c_str(), is_selected))
            {
                current_item = it->first;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        ImGui::EndCombo();
    }

    return current_item;
}

AssetHandle displayTexture2DSelection(const std::string& key, AssetHandle handle)
{
    const auto& data = AssetManager::getMetaData(handle);

    std::string tag = AssetManager::isAssetHandleValid(handle) ? data.name : "No Image";

    const auto& registry = AssetManager::getAssetRegistry();

    AssetHandle current_item = handle;

    if(ImGui::BeginCombo(("Select Image##" + key).c_str(), tag.c_str()))
    {
        // No Selection
        {
            bool is_selected = !AssetManager::isAssetHandleValid(current_item);

            if(ImGui::Selectable("No Image", is_selected))
            {
                current_item = 0;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        for(auto it = registry.begin(); it != registry.end(); ++it)
        {
            if(it->second.type != AssetType::Texture2D) continue;

            bool is_selected = it->first == current_item;

            if(ImGui::Selectable((it->second.name + "##" + std::to_string(it->first)).c_str(), is_selected))
            {
                current_item = it->first;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        ImGui::EndCombo();
    }

    return current_item;
}

AssetHandle displayTexture3DSelection(const std::string& key, AssetHandle handle)
{
    const auto& data = AssetManager::getMetaData(handle);

    std::string tag = AssetManager::isAssetHandleValid(handle) ? data.name : "No Image";

    const auto& registry = AssetManager::getAssetRegistry();

    AssetHandle current_item = handle;

    if(ImGui::BeginCombo(("Select Image##" + key).c_str(), tag.c_str()))
    {
        // No Selection
        {
            bool is_selected = !AssetManager::isAssetHandleValid(current_item);

            if(ImGui::Selectable("No Image", is_selected))
            {
                current_item = 0;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        for(auto it = registry.begin(); it != registry.end(); ++it)
        {
            if(it->second.type != AssetType::Texture3D) continue;

            bool is_selected = it->first == current_item;

            if(ImGui::Selectable((it->second.name + "##" + std::to_string(it->first)).c_str(), is_selected))
            {
                current_item = it->first;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        ImGui::EndCombo();
    }

    return current_item;
}

}    // namespace GUI
}    // namespace atcg