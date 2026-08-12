#pragma once

#include <Core/glm.h>
#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentSerializer.h>
#include <Scene/ComponentRenderer.h>
#include <Material/Medium.h>
#include <Material/PhaseFunction.h>

namespace atcg
{
struct ATCG_API MediumComponent
{
    MediumComponent() = default;
    MediumComponent(AssetHandle medium_handle) : medium_handle(medium_handle) {}
    MediumComponent(const atcg::ref_ptr<Medium>& medium)
    {
        if(AssetManager::isAssetHandleValid(medium->handle))
        {
            medium_handle = medium->handle;
        }
        else
        {
            medium_handle = AssetManager::registerAsset(medium, "medium");
        }
    }

    MediumComponent(AssetHandle medium_handle, AssetHandle phase_function_handle)
        : medium_handle(medium_handle),
          phase_function_handle(phase_function_handle)
    {
    }
    MediumComponent(const atcg::ref_ptr<Medium>& medium, const atcg::ref_ptr<PhaseFunction>& phase_function)
    {
        if(AssetManager::isAssetHandleValid(medium->handle))
        {
            medium_handle = medium->handle;
        }
        else
        {
            medium_handle = AssetManager::registerAsset(medium, "medium");
        }
        if(AssetManager::isAssetHandleValid(phase_function->handle))
        {
            phase_function_handle = phase_function->handle;
        }
        else
        {
            phase_function_handle = AssetManager::registerAsset(phase_function, "phase_function");
        }
    }

    ATCG_INLINE atcg::ref_ptr<Medium> medium() const { return AssetManager::getAsset<Medium>(medium_handle); }
    ATCG_INLINE atcg::ref_ptr<PhaseFunction> phase_function() const
    {
        return AssetManager::getAsset<PhaseFunction>(phase_function_handle);
    }


    AssetHandle medium_handle;
    AssetHandle phase_function_handle;
    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Medium"; }
};

ATCG_DECLARE_COMPONENT_RENDERER(MediumComponent);

namespace Serialization
{
ATCG_DECLARE_COMPONENT_SERIALIZER(MediumComponent);
}

namespace GUI
{
ATCG_DECLARE_COMPONENT_GUI_RENDERER(MediumComponent);
}
}    // namespace atcg