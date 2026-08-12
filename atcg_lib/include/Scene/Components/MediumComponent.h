#pragma once

#include <Core/glm.h>
#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentSerializer.h>
#include <Scene/ComponentRenderer.h>
#include <Material/Medium.h>

namespace atcg
{
struct ATCG_API MediumComponent
{
    MediumComponent() = default;
    MediumComponent(AssetHandle handle) : medium_handle(handle) {}
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

    ATCG_INLINE atcg::ref_ptr<Medium> medium() const { return AssetManager::getAsset<Medium>(medium_handle); }


    AssetHandle medium_handle;
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
}