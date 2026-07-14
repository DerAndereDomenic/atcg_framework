#include <Plugin/Plugin.h>

#include "DiffPathtracingIntegrator.h"

ATCG_PLUGIN_LIBRARY();

extern "C" __declspec(dllexport) void registerPlugin(atcg::PluginRegistry& registry)
{
    registry.registerIntegrator<atcg::DiffPathtracingIntegrator>("DiffPathtracingIntegrator");
}