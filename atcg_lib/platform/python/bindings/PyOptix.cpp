#include "Bindings.h"

#ifdef ATCG_CUDA_BACKEND

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
// Ray tracing contexts, integrators and registries
void bindOptix(BindingContext& ctx)
{
    py::module_ m = ctx.module();

    auto m_raytracing_context_manager = m.def_submodule("RaytracingContextManager");
    auto m_raytracing_context =
        py::class_<atcg::RaytracingContext, atcg::ref_ptr<atcg::RaytracingContext>>(m, "RaytracingContext");

    auto m_integrator = py::class_<atcg::Integrator, atcg::ref_ptr<atcg::Integrator>>(m, "Integrator");
    auto m_path_integrator =
        py::class_<atcg::PathtracingIntegrator, atcg::Integrator, atcg::ref_ptr<atcg::PathtracingIntegrator>>(m,
                                                                                                              "PathInte"
                                                                                                              "grator");
    auto m_volpath_integrator =
        py::class_<atcg::VolPathtracingIntegrator, atcg::Integrator, atcg::ref_ptr<atcg::VolPathtracingIntegrator>>(
            m,
            "VolPathIntegrator");
    auto m_photon_mapping_integrator =
        py::class_<atcg::PhotonMapIntegrator, atcg::Integrator, atcg::ref_ptr<atcg::PhotonMapIntegrator>>(m,
                                                                                                          "PhotonMapInt"
                                                                                                          "egrator");
    auto m_integrator_registry = m.def_submodule("IntegratorRegistry");
    auto m_bsdf_registry       = m.def_submodule("BSDFRegistry");
    auto m_integrator_controller =
        py::class_<atcg::IntegratorController, atcg::ref_ptr<atcg::IntegratorController>>(m, "IntegratorController");

    ctx.define(
        [=]() mutable
        {
            m_raytracing_context_manager.def("createContext", &atcg::RaytracingContextManager::createContext)
                .def("destroyContext", &atcg::RaytracingContextManager::destroyContext);

            m_integrator.def("generateRays",
                             [](const atcg::ref_ptr<atcg::Integrator>& self)
                             {
                                 atcg::Dictionary dict;
                                 self->generateRays(dict);
                                 return dict.getValue<torch::Tensor>("output");
                             });

            m_path_integrator
                .def(py::init(
                    [](const atcg::ref_ptr<atcg::RaytracingContext>& context,
                       const atcg::ref_ptr<atcg::Scene>& scene,
                       const uint32_t width,
                       const uint32_t height)
                    {
                        atcg::Dictionary dict;
                        dict.setValue("scene", scene);
                        dict.setValue("width", width);
                        dict.setValue("height", height);
                        atcg::ref_ptr<atcg::PathtracingIntegrator> integrator =
                            atcg::make_ref<atcg::PathtracingIntegrator>(context, dict);
                        return integrator;
                    }))
                .def("generateRays",
                     [](const atcg::ref_ptr<atcg::PathtracingIntegrator>& self)
                     {
                         atcg::Dictionary dict;
                         self->generateRays(dict);
                         return dict.getValue<torch::Tensor>("output");
                     });

            m_volpath_integrator
                .def(py::init(
                    [](const atcg::ref_ptr<atcg::RaytracingContext>& context,
                       const atcg::ref_ptr<atcg::Scene>& scene,
                       const uint32_t width,
                       const uint32_t height)
                    {
                        atcg::Dictionary dict;
                        dict.setValue("scene", scene);
                        dict.setValue("width", width);
                        dict.setValue("height", height);
                        atcg::ref_ptr<atcg::VolPathtracingIntegrator> integrator =
                            atcg::make_ref<atcg::VolPathtracingIntegrator>(context, dict);
                        return integrator;
                    }))
                .def("generateRays",
                     [](const atcg::ref_ptr<atcg::VolPathtracingIntegrator>& self)
                     {
                         atcg::Dictionary dict;
                         self->generateRays(dict);
                         return dict.getValue<torch::Tensor>("output");
                     });

            m_photon_mapping_integrator
                .def(py::init(
                    [](const atcg::ref_ptr<atcg::RaytracingContext>& context,
                       const atcg::ref_ptr<atcg::Scene>& scene,
                       const uint32_t width,
                       const uint32_t height)
                    {
                        atcg::Dictionary dict;
                        dict.setValue("scene", scene);
                        dict.setValue("width", width);
                        dict.setValue("height", height);
                        atcg::ref_ptr<atcg::PhotonMapIntegrator> integrator =
                            atcg::make_ref<atcg::PhotonMapIntegrator>(context, dict);
                        return integrator;
                    }))
                .def("generateRays",
                     [](const atcg::ref_ptr<atcg::PhotonMapIntegrator>& self)
                     {
                         atcg::Dictionary dict;
                         self->generateRays(dict);
                         return dict.getValue<torch::Tensor>("output");
                     });

            m_integrator_registry.def(
                "createIntegrator",
                [](const std::string& integrator_type,
                   const atcg::ref_ptr<atcg::RaytracingContext>& context,
                   const atcg::ref_ptr<atcg::Scene>& scene,
                   const uint32_t width,
                   const uint32_t height)
                {
                    atcg::Dictionary dict;
                    dict.setValue("scene", scene);
                    dict.setValue("width", width);
                    dict.setValue("height", height);
                    return atcg::IntegratorRegistry::createIntegrator(integrator_type, context, dict);
                });

            m_integrator_registry.def("createIntegrator",
                                      [](const std::string& type,
                                         const atcg::ref_ptr<atcg::RaytracingContext>& context,
                                         const atcg::Dictionary& dict)
                                      { return atcg::IntegratorRegistry::createIntegrator(type, context, dict); });

            m_integrator_controller
                .def(py::init([](const atcg::ref_ptr<atcg::RaytracingContext>& context)
                              { return atcg::make_ref<atcg::IntegratorController>(context); }),
                     "context"_a = nullptr)
                .def("start", &atcg::IntegratorController::start)
                .def("stop", &atcg::IntegratorController::stop)
                .def("isRunning", &atcg::IntegratorController::isRunning)
                .def("setIntegratorType", &atcg::IntegratorController::setIntegratorType, "type"_a)
                .def("getIntegratorType", &atcg::IntegratorController::getIntegratorType)
                .def("requestRebuild", &atcg::IntegratorController::requestRebuild)
                .def("reset", &atcg::IntegratorController::reset)
                .def("release", &atcg::IntegratorController::release)
                .def("getIntegrator", &atcg::IntegratorController::getIntegrator)
                .def("render", &atcg::IntegratorController::render, "scene"_a, "target"_a)
                .def("getOutputTexture", &atcg::IntegratorController::getOutputTexture)
                .def("getOutputEntityTexture", &atcg::IntegratorController::getOutputEntityTexture);
        });
}
}    // namespace python
}    // namespace atcg

#endif
