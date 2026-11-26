#include "Bindings.h"

#ifdef ATCG_CUDA_BACKEND

    #include <BSDF/BSDFRegistry.h>
    #include <Medium/Medium.h>
    #include <Medium/PhaseFunction.h>
    #include <Scene/OptixScene.h>
    #include <Shape/ShapeSampler.h>

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
namespace
{
/**
 * @brief Create an integrator from a scene and an image size
 *
 * @tparam IntegratorClass The integrator class
 * @param context The raytracing context
 * @param scene The scene
 * @param width The width
 * @param height The height
 *
 * @return The integrator
 */
template<typename IntegratorClass>
atcg::ref_ptr<IntegratorClass> createIntegratorFromScene(const atcg::ref_ptr<atcg::RaytracingContext>& context,
                                                         const atcg::ref_ptr<atcg::Scene>& scene,
                                                         const uint32_t width,
                                                         const uint32_t height)
{
    atcg::Dictionary dict;
    dict.setValue("scene", scene);
    dict.setValue("width", width);
    dict.setValue("height", height);
    return atcg::make_ref<IntegratorClass>(context, dict);
}

/**
 * @brief Get the registered types of a registry that is stored in the system registry
 *
 * @tparam Registry The registry type
 * @param registry_name The name of the registry for the log
 *
 * @return The registered types
 */
template<typename Registry>
std::vector<std::string> getRegisteredTypes(const char* registry_name)
{
    if(!atcg::SystemRegistry::instance()->hasSystem<Registry>())
    {
        ATCG_WARN("{0} not found", registry_name);
        return {};
    }

    return atcg::SystemRegistry::instance()->getSystem<Registry>()->getRegisteredTypes();
}
}    // namespace

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

    // The settings specializations are not exported from atcg_lib. They can't be constructed here, use
    // IntegratorRegistry.createIntegratorSettings or IntegratorController.getSettings instead.
    auto m_integrator_settings =
        py::class_<atcg::IntegratorSettingsBase, atcg::ref_ptr<atcg::IntegratorSettingsBase>>(m, "IntegratorSettings");
    auto m_path_integrator_settings =
        py::class_<atcg::IntegratorSettings<atcg::PathtracingIntegrator>,
                   atcg::IntegratorSettingsBase,
                   atcg::ref_ptr<atcg::IntegratorSettings<atcg::PathtracingIntegrator>>>(m, "PathIntegratorSettings");
    auto m_volpath_integrator_settings =
        py::class_<atcg::IntegratorSettings<atcg::VolPathtracingIntegrator>,
                   atcg::IntegratorSettingsBase,
                   atcg::ref_ptr<atcg::IntegratorSettings<atcg::VolPathtracingIntegrator>>>(m,
                                                                                            "VolPathIntegratorSetting"
                                                                                            "s");
    auto m_photon_mapping_integrator_settings =
        py::class_<atcg::IntegratorSettings<atcg::PhotonMapIntegrator>,
                   atcg::IntegratorSettingsBase,
                   atcg::ref_ptr<atcg::IntegratorSettings<atcg::PhotonMapIntegrator>>>(m,
                                                                                       "PhotonMapIntegratorSettings");

    auto m_integrator_registry = m.def_submodule("IntegratorRegistry");
    auto m_bsdf_registry       = m.def_submodule("BSDFRegistry");
    auto m_integrator_controller =
        py::class_<atcg::IntegratorController, atcg::ref_ptr<atcg::IntegratorController>>(m, "IntegratorController");

    // Optix components
    auto m_bsdf_component_type = py::enum_<atcg::BSDFComponentType>(m, "BSDFComponentType", py::arithmetic());
    auto m_emitter_flags       = py::enum_<atcg::EmitterFlags>(m, "EmitterFlags", py::arithmetic());

    auto m_optix_component = py::class_<atcg::OptixComponent, atcg::ref_ptr<atcg::OptixComponent>>(m, "OptixComponent");
    auto m_bsdf            = py::class_<atcg::BSDF, atcg::OptixComponent, atcg::ref_ptr<atcg::BSDF>>(m, "BSDF");
    auto m_emitter = py::class_<atcg::Emitter, atcg::OptixComponent, atcg::ref_ptr<atcg::Emitter>>(m, "Emitter");
    auto m_environment_emitter =
        py::class_<atcg::EnvironmentEmitter, atcg::Emitter, atcg::ref_ptr<atcg::EnvironmentEmitter>>(m,
                                                                                                     "EnvironmentEmitt"
                                                                                                     "er");
    auto m_medium = py::class_<atcg::Medium, atcg::OptixComponent, atcg::ref_ptr<atcg::Medium>>(m, "Medium");
    auto m_phase_function =
        py::class_<atcg::PhaseFunction, atcg::OptixComponent, atcg::ref_ptr<atcg::PhaseFunction>>(m, "PhaseFunction");
    auto m_shape = py::class_<atcg::Shape, atcg::OptixComponent, atcg::ref_ptr<atcg::Shape>>(m, "Shape");
    auto m_shape_sampler =
        py::class_<atcg::ShapeSampler, atcg::OptixComponent, atcg::ref_ptr<atcg::ShapeSampler>>(m, "ShapeSampler");
    auto m_shape_instance =
        py::class_<atcg::ShapeInstance, atcg::OptixComponent, atcg::ref_ptr<atcg::ShapeInstance>>(m, "ShapeInstance");
    auto m_film   = py::class_<atcg::Film, atcg::OptixComponent, atcg::ref_ptr<atcg::Film>>(m, "Film");
    auto m_sensor = py::class_<atcg::Sensor, atcg::OptixComponent, atcg::ref_ptr<atcg::Sensor>>(m, "Sensor");
    auto m_ias =
        py::class_<atcg::InstanceAccelerationStructure, atcg::ref_ptr<atcg::InstanceAccelerationStructure>>(m,
                                                                                                            "InstanceAc"
                                                                                                            "celeration"
                                                                                                            "Structur"
                                                                                                            "e");

    // Optix scene and the components of its entities
    auto m_optix_scene    = py::class_<atcg::OptixScene, atcg::Scene, atcg::ref_ptr<atcg::OptixScene>>(m, "OptixScene");
    auto m_bsdf_component = py::class_<atcg::BSDFComponent>(m, "BSDFComponent");
    auto m_emitter_component        = py::class_<atcg::EmitterComponent>(m, "EmitterComponent");
    auto m_medium_component         = py::class_<atcg::MediumComponent>(m, "MediumComponent");
    auto m_phase_function_component = py::class_<atcg::PhaseFunctionComponent>(m, "PhaseFunctionComponent");
    auto m_shape_component          = py::class_<atcg::ShapeComponent>(m, "ShapeComponent");

    ctx.define(
        [=]() mutable
        {
            m_raytracing_context.def_static("initRaytracingAPI", &atcg::RaytracingContext::initRaytracingAPI)
                .def("getContextHandle",
                     [](const atcg::RaytracingContext& self)
                     { return reinterpret_cast<uintptr_t>(self.getContextHandle()); });

            m_raytracing_context_manager
                .def("createContext", &atcg::RaytracingContextManager::createContext, "device_id"_a = 0)
                .def(
                    "destroyContext",
                    [](atcg::ref_ptr<atcg::RaytracingContext> context)
                    {
                        if(!context)
                        {
                            ATCG_WARN("RaytracingContextManager.destroyContext: context is None");
                            return;
                        }
                        atcg::RaytracingContextManager::destroyContext(context);
                    },
                    "context"_a,
                    "Destroys the OptiX context. The python object can't be used afterwards, drop all references to "
                    "it.");

            m_integrator
                .def("generateRays",
                     [](const atcg::ref_ptr<atcg::Integrator>& self)
                     {
                         atcg::Dictionary dict;
                         self->generateRays(dict);
                         return dict.getValue<torch::Tensor>("output");
                     })
                .def(
                    "generateRays",
                    [](const atcg::ref_ptr<atcg::Integrator>& self, atcg::Dictionary& dict) -> atcg::Dictionary&
                    {
                        self->generateRays(dict);
                        return dict;
                    },
                    "dict"_a,
                    py::return_value_policy::reference,
                    "Generates the rays with the given input dictionary. The dictionary is modified in place and "
                    "returned.")
                .def("reset", &atcg::Integrator::reset)
    #ifndef ATCG_HEADLESS
                .def("onImGuiRender", &atcg::Integrator::onImGuiRender)
    #endif
                ;

            m_path_integrator
                .def(py::init(&createIntegratorFromScene<atcg::PathtracingIntegrator>),
                     "context"_a,
                     "scene"_a,
                     "width"_a,
                     "height"_a)
                .def(py::init<const atcg::ref_ptr<atcg::RaytracingContext>&, const atcg::Dictionary&>(),
                     "context"_a,
                     "dict"_a)
                .def_static("getType", &atcg::PathtracingIntegrator::getType);

            m_volpath_integrator
                .def(py::init(&createIntegratorFromScene<atcg::VolPathtracingIntegrator>),
                     "context"_a,
                     "scene"_a,
                     "width"_a,
                     "height"_a)
                .def(py::init<const atcg::ref_ptr<atcg::RaytracingContext>&, const atcg::Dictionary&>(),
                     "context"_a,
                     "dict"_a)
                .def_static("getType", &atcg::VolPathtracingIntegrator::getType);

            m_photon_mapping_integrator
                .def(py::init(&createIntegratorFromScene<atcg::PhotonMapIntegrator>),
                     "context"_a,
                     "scene"_a,
                     "width"_a,
                     "height"_a)
                .def(py::init<const atcg::ref_ptr<atcg::RaytracingContext>&, const atcg::Dictionary&>(),
                     "context"_a,
                     "dict"_a)
                .def_static("getType", &atcg::PhotonMapIntegrator::getType);

            // Integrator settings
            m_integrator_settings.def("getType", &atcg::IntegratorSettingsBase::getType)
                .def("toDictionary", &atcg::IntegratorSettingsBase::toDictionary)
                .def("toJson", [](const atcg::IntegratorSettingsBase& self) { return self.toJson().dump(); })
                .def(
                    "fromJson",
                    [](atcg::IntegratorSettingsBase& self, const std::string& json)
                    {
                        nlohmann::json j = nlohmann::json::parse(json, nullptr, false);
                        if(j.is_discarded())
                        {
                            ATCG_ERROR("IntegratorSettings.fromJson: invalid json");
                            return;
                        }
                        self.fromJson(j);
                    },
                    "json"_a)
    #ifndef ATCG_HEADLESS
                .def("renderSettings", &atcg::IntegratorSettingsBase::renderSettings)
    #endif
                ;

            m_path_integrator_settings
                .def_readwrite("use_rr", &atcg::IntegratorSettings<atcg::PathtracingIntegrator>::_use_rr)
                .def_readwrite("max_depth", &atcg::IntegratorSettings<atcg::PathtracingIntegrator>::_max_depth)
                .def_readwrite("hide_emitters", &atcg::IntegratorSettings<atcg::PathtracingIntegrator>::_hide_emitters);

            m_volpath_integrator_settings
                .def_readwrite("use_rr", &atcg::IntegratorSettings<atcg::VolPathtracingIntegrator>::_use_rr)
                .def_readwrite("max_depth", &atcg::IntegratorSettings<atcg::VolPathtracingIntegrator>::_max_depth)
                .def_readwrite("hide_emitters",
                               &atcg::IntegratorSettings<atcg::VolPathtracingIntegrator>::_hide_emitters);

            m_photon_mapping_integrator_settings
                .def_readwrite("reduction_factor",
                               &atcg::IntegratorSettings<atcg::PhotonMapIntegrator>::_reduction_factor)
                .def_readwrite("gather_radius", &atcg::IntegratorSettings<atcg::PhotonMapIntegrator>::_gather_radius)
                .def_readwrite("max_photon_depth",
                               &atcg::IntegratorSettings<atcg::PhotonMapIntegrator>::_max_photon_depth)
                .def_readwrite("max_camera_depth",
                               &atcg::IntegratorSettings<atcg::PhotonMapIntegrator>::_max_camera_depth)
                .def_readwrite("max_num_photons",
                               &atcg::IntegratorSettings<atcg::PhotonMapIntegrator>::_max_num_photons);

            // Registries
            m_integrator_registry
                .def(
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
                    },
                    "type"_a,
                    "context"_a,
                    "scene"_a,
                    "width"_a,
                    "height"_a)
                .def(
                    "createIntegrator",
                    [](const std::string& type,
                       const atcg::ref_ptr<atcg::RaytracingContext>& context,
                       const atcg::Dictionary& dict)
                    { return atcg::IntegratorRegistry::createIntegrator(type, context, dict); },
                    "type"_a,
                    "context"_a,
                    "dict"_a)
                .def(
                    "createIntegratorSettings",
                    [](const std::string& type) { return atcg::IntegratorRegistry::createIntegratorSettings(type); },
                    "type"_a)
                .def("getRegisteredTypes",
                     []() { return getRegisteredTypes<atcg::IntegratorRegistry::Registry>("Integrator registry"); });

            m_bsdf_registry.def("validateMaterialCoverage", &atcg::BSDFRegistry::validateMaterialCoverage)
                .def("getRegisteredTypes",
                     []() { return getRegisteredTypes<atcg::BSDFRegistry::Registry>("BSDF registry"); });

            m_integrator_controller
                .def(py::init([](const atcg::ref_ptr<atcg::RaytracingContext>& context)
                              { return atcg::make_ref<atcg::IntegratorController>(context); }),
                     "context"_a = nullptr)
                .def("start", &atcg::IntegratorController::start)
                .def("stop", &atcg::IntegratorController::stop)
                .def("isRunning", &atcg::IntegratorController::isRunning)
                .def("setIntegratorType", &atcg::IntegratorController::setIntegratorType, "type"_a)
                .def("getIntegratorType", &atcg::IntegratorController::getIntegratorType)
                .def("getSettings", &atcg::IntegratorController::getSettings)
                .def("requestRebuild", &atcg::IntegratorController::requestRebuild)
                .def("reset", &atcg::IntegratorController::reset)
                .def("release", &atcg::IntegratorController::release)
                .def("getIntegrator", &atcg::IntegratorController::getIntegrator)
                .def("render", &atcg::IntegratorController::render, "scene"_a, "target"_a)
                .def("getOutputTexture", &atcg::IntegratorController::getOutputTexture)
                .def("getOutputEntityTexture", &atcg::IntegratorController::getOutputEntityTexture);

            // Optix components
            m_bsdf_component_type.value("None", atcg::BSDFComponentType::None)
                .value("IdealReflection", atcg::BSDFComponentType::IdealReflection)
                .value("GlossyReflection", atcg::BSDFComponentType::GlossyReflection)
                .value("DiffuseReflection", atcg::BSDFComponentType::DiffuseReflection)
                .value("IdealTransmission", atcg::BSDFComponentType::IdealTransmission)
                .value("GlossyTransmission", atcg::BSDFComponentType::GlossyTransmission)
                .value("DiffuseTransmission", atcg::BSDFComponentType::DiffuseTransmission)
                .value("NullTransmission", atcg::BSDFComponentType::NullTransmission)
                .value("AnyDelta", atcg::BSDFComponentType::AnyDelta)
                .value("AnyReflection", atcg::BSDFComponentType::AnyReflection)
                .value("AnyTransmission", atcg::BSDFComponentType::AnyTransmission)
                .value("Any", atcg::BSDFComponentType::Any);

            m_emitter_flags.value("None", atcg::EmitterFlags::None)
                .value("DistantEmitter", atcg::EmitterFlags::DistantEmitter)
                .value("InfinitesimalSize", atcg::EmitterFlags::InfinitesimalSize);

            m_optix_component.def("isInitialized", &atcg::OptixComponent::isInitialized);
    #ifndef ATCG_HEADLESS
            m_optix_component.def("onImGuiRender", &atcg::OptixComponent::onImGuiRender);
    #endif


            m_bsdf.def("flags", &atcg::BSDF::flags);

            m_emitter.def("flags", &atcg::Emitter::flags);

            m_medium.def("getPhaseFunction", &atcg::Medium::getPhaseFunction);

            m_shape.def("getShapeType", &atcg::Shape::getShapeType);

            m_shape_instance.def("getShape", &atcg::ShapeInstance::getShape)
                .def("getBSDF", &atcg::ShapeInstance::getBSDF)
                .def("getEmitter", &atcg::ShapeInstance::getEmitter)
                .def("getInsideMedium", &atcg::ShapeInstance::getInsideMedium)
                .def("getOutsideMedium", &atcg::ShapeInstance::getOutsideMedium)
                .def("getSampler", &atcg::ShapeInstance::getSampler)
                .def("getTransform", &atcg::ShapeInstance::getTransform)
                .def("entity_id", &atcg::ShapeInstance::entity_id)
                .def("color", &atcg::ShapeInstance::color);

            m_film.def("getWidth", &atcg::Film::getWidth)
                .def("getHeight", &atcg::Film::getHeight)
                .def("resize", &atcg::Film::resize, "width"_a, "height"_a)
                .def("develop", &atcg::Film::develop)
                .def("clear", &atcg::Film::clear);

            m_sensor.def("getFilm", &atcg::Sensor::getFilm)
                .def("getCamera", &atcg::Sensor::getCamera)
                .def("markDirty", &atcg::Sensor::markDirty);

            // Optix scene and the components of its entities
            m_optix_scene.def("getShapes", &atcg::OptixScene::getShapes)
                .def("getEmitter", &atcg::OptixScene::getEmitter)
                .def("getEnvironmentEmitter", &atcg::OptixScene::getEnvironmentEmitter)
                .def("getIAS", &atcg::OptixScene::getIAS)
                .def("getSensor", &atcg::OptixScene::getSensor)
                .def("setSensor", &atcg::OptixScene::setSensor, "sensor"_a);

            m_bsdf_component.def_readonly("bsdf", &atcg::BSDFComponent::bsdf);
            m_emitter_component.def_readonly("emitter", &atcg::EmitterComponent::emitter);
            m_medium_component.def_readonly("medium", &atcg::MediumComponent::medium);
            m_phase_function_component.def_readonly("phase_function", &atcg::PhaseFunctionComponent::phase_function);
            m_shape_component.def_readonly("shape", &atcg::ShapeComponent::shape);
        });
}
}    // namespace python
}    // namespace atcg

#endif
