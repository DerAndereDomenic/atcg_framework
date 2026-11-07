#include <stdio.h>
#include <Plugin/Plugin.h>

#include "PerfectDielectric.h"
#include "PerfectDielectricBSDF.h"

#include <ATCG.h>

#include <Core/Path.h>
#include <Core/Assert.h>
#include <Core/CUDA.h>
#include <Core/Common.h>
#include <Scene/Components.h>
#include <Scene/Entity.h>
#include <DataStructure/WorkerPool.h>

#ifdef ATCG_CUDA_BACKEND
    #include <Shape/Shape.h>
    #include <Shape/ShapeInstance.h>
    #include <Shape/MeshShape.h>
    #include <Emitter/MeshEmitter.h>
    #include <Scene/SceneAdapter.h>
#endif

#ifndef ATCG_HEADLESS
    #include <imgui.h>
#endif

ATCG_PLUGIN_LIBRARY();

extern "C" ATCG_EXPORT void registerPlugin(atcg::PluginRegistry& registry)
{
    registry.registerMaterial<atcg::PerfectDielectric>("PerfectDielectric");
    registry.registerBSDF<atcg::PerfectDielectricBSDF>("PerfectDielectric");
}

extern "C" ATCG_EXPORT void registerPythonBindings(pybind11::module& m) {}