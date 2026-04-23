#pragma once

#include <Asset/AssetManagerSystem.h>
#include <DataStructure/Graph.h>

namespace atcg
{
struct GeometryComponent
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

    AssetHandle graph_handle;

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Geometry"; }
};
}    // namespace atcg