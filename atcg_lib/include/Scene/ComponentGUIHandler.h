#pragma once

#include <Scene/Scene.h>

#include <Renderer/Framebuffer.h>
#include <Scene/RevisionStack.h>

#ifndef ATCG_HEADLESS
    #include <imgui.h>
#endif

namespace atcg
{
namespace GUI
{

/**
 * @brief A class that handles the rendering of gui components
 *
 * To add custom rendering code, create a class that specializes this class and add the rendering code for the custom
 * component.
 *
 * @code{.cpp}
 * template<>
 * struct atcg::GUI::ComponentGUIRenderer<CustomComponent>
 * {
 *     void draw_component(const atcg::ref_ptr<Scene>& scene, Entity entity, CustomComponent& component) const
 *     {
 *         // Render Code
 *     }
 * };
 * @endcode
 */
template<typename T>
struct ComponentGUIRenderer
{
    /**
     * @brief Draw the component
     *
     * @param scene The scene the entity belongs to
     * @param entity The entity that holds the component
     * @param component The component to render
     */
    void draw_component(const atcg::ref_ptr<Scene>& scene, Entity entity, T& component) const {}
};

template<typename T>
struct is_gui_addable : std::true_type
{
};

#define ATCG_DECLARE_COMPONENT_GUI_RENDERER(ComponentType)                                                             \
    template<>                                                                                                         \
    struct ComponentGUIRenderer<ComponentType>                                                                         \
    {                                                                                                                  \
        void draw_component(const atcg::ref_ptr<Scene>& scene, Entity entity, ComponentType& component) const;         \
    }


template<typename T>
ATCG_INLINE void drawComponent(const atcg::ref_ptr<Scene>& scene, Entity entity)
{
#ifndef ATCG_HEADLESS
    const ImGuiTreeNodeFlags treeNodeFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed |
                                             ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap |
                                             ImGuiTreeNodeFlags_FramePadding;

    if(entity.hasComponent<T>())
    {
        auto& component = entity.getComponent<T>();
        bool open       = ImGui::TreeNodeEx((void*)typeid(T).hash_code(), treeNodeFlags, T::toString());

        ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();
        ImGui::SameLine(contentRegionAvailable.x);
        if(ImGui::Button("+"))
        {
            ImGui::OpenPopup("ComponentSettings");
        }

        bool removeComponent = false;
        if(ImGui::BeginPopup("ComponentSettings"))
        {
            if(ImGui::MenuItem("Remove component")) removeComponent = true;

            ImGui::EndPopup();
        }

        if(open)
        {
            ComponentGUIRenderer<T>().draw_component(scene, entity, component);
            ImGui::TreePop();
        }

        if(removeComponent)
        {
            atcg::RevisionStack::startRecording<ComponentRemovedRevision<T>>(scene, entity);
            entity.removeComponent<T>();
            atcg::RevisionStack::endRecording();
        }
    }
#endif
}

template<typename T>
ATCG_INLINE void displayAddComponentEntry(const atcg::ref_ptr<atcg::Scene>& scene, Entity entity)
{
#ifndef ATCG_HEADLESS
    if constexpr(!is_gui_addable<T>::value) return;

    if(!entity.hasComponent<T>())
    {
        if(ImGui::MenuItem(T::toString()))
        {
            atcg::RevisionStack::startRecording<ComponentAddedRevision<T>>(scene, entity);
            entity.addComponent<T>();
            ImGui::CloseCurrentPopup();
            atcg::RevisionStack::endRecording();
        }
    }
#endif
}

AssetHandle displayMaterialSelection(const std::string& key, AssetHandle handle);

AssetHandle displayGraphSelection(const std::string& key, AssetHandle handle);

AssetHandle displayScriptSelection(const std::string& key, AssetHandle handle);

AssetHandle displayShaderSelection(const std::string& key, AssetHandle handle);

AssetHandle displayTexture2DSelection(const std::string& key, AssetHandle handle);

AssetHandle displayTexture3DSelection(const std::string& key, AssetHandle handle);
}    // namespace GUI
}    // namespace atcg