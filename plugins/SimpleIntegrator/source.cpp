#include <stdio.h>
#include <Plugin/Plugin.h>

#include "BSDFIntegrator.h"
#include "EmitterIntegrator.h"

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
#ifdef ATCG_CUDA_BACKEND
    registry.registerIntegrator<atcg::BSDFIntegrator>("BSDFIntegrator");
    registry.registerIntegrator<atcg::EmitterIntegrator>("EmitterIntegrator");
#endif
}

extern "C" ATCG_EXPORT void registerPythonBindings(pybind11::module& m) {}