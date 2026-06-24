#pragma once

#include <Asset/AssetManagerSystem.h>
#include <DataStructure/Graph.h>
#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentSerializer.h>
#include <Scene/ComponentRenderer.h>

namespace atcg
{
struct ATCG_API GeometryComponent
{
    GeometryComponent() = default;
    GeometryComponent(AssetHandle handle) : graph_handle(handle) {}
    GeometryComponent(const atcg::ref_ptr<Graph>& graph)
    {
        if(AssetManager::isAssetHandleValid(graph->handle))
        {
            graph_handle = graph->handle;
        }
        else
        {
            graph_handle = AssetManager::registerAsset(graph, "graph");
        }
    }

    ATCG_INLINE atcg::ref_ptr<Graph> graph() const { return AssetManager::getAsset<Graph>(graph_handle); }

    bool draw_bounding_box = false;

    AssetHandle graph_handle;

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Geometry"; }
};

ATCG_DECLARE_COMPONENT_RENDERER(GeometryComponent);

namespace Serialization
{
ATCG_DECLARE_COMPONENT_SERIALIZER(GeometryComponent);
}

namespace GUI
{
ATCG_DECLARE_COMPONENT_GUI_RENDERER(GeometryComponent);
}
}    // namespace atcg