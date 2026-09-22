#include <Plugin/Plugin.h>

#include "VolDiffPathtracingIntegrator.h"
#include <torch/python.h>

ATCG_PLUGIN_LIBRARY();

extern "C" __declspec(dllexport) void registerPlugin(atcg::PluginRegistry& registry)
{
    registry.registerIntegrator<atcg::VolDiffPathtracingIntegrator>("VolDiffPathtracingIntegrator");
}

extern "C" __declspec(dllexport) void registerPythonBindings(pybind11::module& m)
{
    pybind11::class_<atcg::VolDiffPathtracingIntegrator,
                     atcg::Integrator,
                     atcg::ref_ptr<atcg::VolDiffPathtracingIntegrator>>(m, "VolDiffPathtracingIntegrator")
        .def(pybind11::init(
            [](const atcg::ref_ptr<atcg::RaytracingContext>& context,
               const atcg::ref_ptr<atcg::Scene>& scene,
               const uint32_t width,
               const uint32_t height)
            {
                atcg::Dictionary dict;
                dict.setValue("scene", scene);
                dict.setValue("width", width);
                dict.setValue("height", height);
                atcg::ref_ptr<atcg::VolDiffPathtracingIntegrator> integrator =
                    atcg::make_ref<atcg::VolDiffPathtracingIntegrator>(context, dict);
                return integrator;
            }))
        .def("generateRays",
             [](const atcg::ref_ptr<atcg::VolDiffPathtracingIntegrator>& self)
             {
                 atcg::Dictionary dict;
                 self->generateRays(dict);
                 torch::Tensor output = dict.getValue<torch::Tensor>("output_img");
                 return output;
             })
        .def("generateRays",
             [](const atcg::ref_ptr<atcg::VolDiffPathtracingIntegrator>& self, uint32_t rng_index)
             {
                 atcg::Dictionary dict;
                 dict.setValue("rng_index", rng_index);
                 self->generateRays(dict);
                 torch::Tensor output = dict.getValue<torch::Tensor>("output_img");
                 return output;
             })
        .def("getOptixScene",
             [](const atcg::ref_ptr<atcg::VolDiffPathtracingIntegrator>& self)
             {
                 auto optix_scene = self->getDictionary().getValue<atcg::ref_ptr<atcg::OptixScene>>("optix_scene");
                 return optix_scene;
             })
        .def("forwardTrace", &atcg::VolDiffPathtracingIntegrator::forwardTrace)
        .def("backwardTrace", &atcg::VolDiffPathtracingIntegrator::backwardTrace)
        .def("getAOVBuffer", &atcg::VolDiffPathtracingIntegrator::getAOVBuffer);
}