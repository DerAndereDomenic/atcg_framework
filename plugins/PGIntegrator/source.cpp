#include <stdio.h>
#include <Plugin/Plugin.h>

#include "PGIntegrator.h"

#include <ATCG.h>

#include <Core/Path.h>
#include <Core/Assert.h>
#include <Core/CUDA.h>
#include <Core/Common.h>
#include <Scene/Components.h>
#include <Scene/Entity.h>
#include <DataStructure/WorkerPool.h>

#ifndef ATCG_HEADLESS
    #include <imgui.h>
#endif


ATCG_PLUGIN_LIBRARY();

extern "C" ATCG_EXPORT void registerPlugin(atcg::PluginRegistry& registry)
{
#ifdef ATCG_CUDA_BACKEND
    registry.registerIntegrator<atcg::PGIntegrator>("PGIntegrator");
#endif
}

extern "C" ATCG_EXPORT void registerPythonBindings(pybind11::module& m) {}