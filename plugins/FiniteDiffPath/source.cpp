#include <Plugin/Plugin.h>

#include "FiniteDiffPathIntegrator.h"
#include <torch/python.h>
#include <Scene/OptixScene.h>

ATCG_PLUGIN_LIBRARY();

extern "C" __declspec(dllexport) void registerPlugin(atcg::PluginRegistry& registry)
{
    registry.registerIntegrator<atcg::FiniteDiffPathtracingIntegrator>("FiniteDiffPathIntegrator");
}

extern "C" __declspec(dllexport) void registerPythonBindings(pybind11::module& m)
{
    pybind11::class_<atcg::FiniteDiffPathtracingIntegrator,
                     atcg::Integrator,
                     atcg::ref_ptr<atcg::FiniteDiffPathtracingIntegrator>>(m, "FiniteDiffPathIntegrator")
        .def(pybind11::init(
            [](const atcg::ref_ptr<atcg::RaytracingContext>& context,
               const atcg::ref_ptr<atcg::Scene>& scene,
               const uint32_t width,
               const uint32_t height,
               const atcg::ref_ptr<atcg::Integrator>& integrator)
            {
                atcg::Dictionary dict;
                dict.setValue("scene", scene);
                dict.setValue("width", width);
                dict.setValue("height", height);
                dict.setValue("integrator", integrator);
                atcg::ref_ptr<atcg::FiniteDiffPathtracingIntegrator> finite_integrator =
                    atcg::make_ref<atcg::FiniteDiffPathtracingIntegrator>(context, dict);
                return finite_integrator;
            }))
        .def("generateRays",
             [](const atcg::ref_ptr<atcg::FiniteDiffPathtracingIntegrator>& self)
             {
                 atcg::Dictionary dict;
                 self->generateRays(dict);
                 torch::Tensor output = dict.getValue<torch::Tensor>("output_img");
                 return output;
             })
        .def("generateRays",
             [](const atcg::ref_ptr<atcg::FiniteDiffPathtracingIntegrator>& self, uint32_t rng_index)
             {
                 atcg::Dictionary dict;
                 dict.setValue("rng_index", rng_index);
                 self->generateRays(dict);
                 torch::Tensor output = dict.getValue<torch::Tensor>("output_img");
                 return output;
             })
        .def("getOptixScene",
             [](const atcg::ref_ptr<atcg::FiniteDiffPathtracingIntegrator>& self)
             {
                 auto base_integrator = self->getIntegrator();
                 auto optix_scene = base_integrator->getDictionary().getValue<atcg::ref_ptr<atcg::OptixScene>>("optix_"
                                                                                                               "scene");
                 return optix_scene;
             });
}