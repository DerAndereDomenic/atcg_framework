#include <Plugin/Plugin.h>

#include "AttachedDiffPathtracingIntegrator.h"

ATCG_PLUGIN_LIBRARY();

extern "C" __declspec(dllexport) void registerPlugin(atcg::PluginRegistry& registry)
{
    registry.registerIntegrator<atcg::AttachedDiffPathtracingIntegrator>("AttachedDiffPathtracingIntegrator");
}