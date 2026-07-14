#include <Plugin/Plugin.h>

#include "FiniteDiffPathIntegrator.h"

ATCG_PLUGIN_LIBRARY();

extern "C" __declspec(dllexport) void registerPlugin(atcg::PluginRegistry& registry)
{
    registry.registerIntegrator<atcg::FiniteDiffPathtracingIntegrator>("FiniteDiffPathIntegrator");
}