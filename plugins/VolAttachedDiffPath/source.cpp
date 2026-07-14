#include <Plugin/Plugin.h>

#include "VolAttachedDiffPathtracingIntegrator.h"

ATCG_PLUGIN_LIBRARY();

extern "C" __declspec(dllexport) void registerPlugin(atcg::PluginRegistry& registry)
{
    registry.registerIntegrator<atcg::VolAttachedDiffPathtracingIntegrator>("VolAttachedDiffPathtracingIntegrator");
}