#include "Bindings.h"

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
// Materials and the material registry
void bindMaterial(BindingContext& ctx)
{
    py::module_ m = ctx.module();

    auto m_material = py::class_<atcg::Material, atcg::Asset, atcg::ref_ptr<atcg::Material>>(m, "Material");
    auto m_opaque_material =
        py::class_<atcg::OpaqueMaterial, atcg::Material, atcg::ref_ptr<atcg::OpaqueMaterial>>(m, "OpaqueMaterial");
    auto m_dielectric_material =
        py::class_<atcg::DielectricMaterial, atcg::Material, atcg::ref_ptr<atcg::DielectricMaterial>>(m,
                                                                                                      "DielectricMate"
                                                                                                      "rial");
    auto m_null_material =
        py::class_<atcg::NullMaterial, atcg::Material, atcg::ref_ptr<atcg::NullMaterial>>(m, "NullMaterial");

    auto m_material_registry = m.def_submodule("MaterialRegistry");

    ctx.define(
        [=]() mutable
        {
            m_material
                .def("asOpaque",
                     [](const atcg::ref_ptr<atcg::Material>& self) -> atcg::ref_ptr<atcg::OpaqueMaterial>
                     {
                         auto ptr = std::dynamic_pointer_cast<atcg::OpaqueMaterial>(self);
                         if(!ptr) ATCG_WARN("Material is not an OpaqueMaterial");
                         return ptr;
                     })
                .def("asDielectric",
                     [](const atcg::ref_ptr<atcg::Material>& self) -> atcg::ref_ptr<atcg::DielectricMaterial>
                     {
                         auto ptr = std::dynamic_pointer_cast<atcg::DielectricMaterial>(self);
                         if(!ptr) ATCG_WARN("Material is not a DielectricMaterial");
                         return ptr;
                     })
                .def("asNull",
                     [](const atcg::ref_ptr<atcg::Material>& self) -> atcg::ref_ptr<atcg::NullMaterial>
                     {
                         auto ptr = std::dynamic_pointer_cast<atcg::NullMaterial>(self);
                         if(!ptr) ATCG_WARN("Material is not a NullMaterial");
                         return ptr;
                     })
                .def("getMaterialType", &atcg::Material::getMaterialType)
                .def("clone", &atcg::Material::clone)
                .def_static("getStaticType", &atcg::Material::getStaticType);

            m_opaque_material.attr("TypeName")     = py::str(atcg::OpaqueMaterial::TypeName);
            m_dielectric_material.attr("TypeName") = py::str(atcg::DielectricMaterial::TypeName);
            m_null_material.attr("TypeName")       = py::str(atcg::NullMaterial::TypeName);

            m_opaque_material.def(py::init<>())
                .def("getDiffuseTexture", &atcg::OpaqueMaterial::getDiffuseTexture)
                .def("getNormalTexture", &atcg::OpaqueMaterial::getNormalTexture)
                .def("getRoughnessTexture", &atcg::OpaqueMaterial::getRoughnessTexture)
                .def("getMetallicTexture", &atcg::OpaqueMaterial::getMetallicTexture)
                .def("getIorTexture", &atcg::OpaqueMaterial::getIorTexture)
                .def("setDiffuseTexture", &atcg::OpaqueMaterial::setDiffuseTexture)
                .def("setNormalTexture", &atcg::OpaqueMaterial::setNormalTexture)
                .def("setRoughnessTexture", &atcg::OpaqueMaterial::setRoughnessTexture)
                .def("setMetallicTexture", &atcg::OpaqueMaterial::setMetallicTexture)
                .def("setIorTexture", &atcg::OpaqueMaterial::setIorTexture)
                .def("setDiffuseColor",
                     [](atcg::OpaqueMaterial& material, const glm::vec3& color) { material.setDiffuseColor(color); })
                .def("setDiffuseColor",
                     [](atcg::OpaqueMaterial& material, const glm::vec4& color) { material.setDiffuseColor(color); })
                .def("setRoughness", &atcg::OpaqueMaterial::setRoughness)
                .def("setMetallic", &atcg::OpaqueMaterial::setMetallic)
                .def("setIor", &atcg::OpaqueMaterial::setIor)
                .def("removeNormalMap", &atcg::OpaqueMaterial::removeNormalMap);

            m_dielectric_material.def(py::init<>())
                .def("getDiffuseTexture", &atcg::DielectricMaterial::getDiffuseTexture)
                .def("getRoughnessTexture", &atcg::DielectricMaterial::getRoughnessTexture)
                .def("getIorTexture", &atcg::DielectricMaterial::getIorTexture)
                .def("setDiffuseTexture", &atcg::DielectricMaterial::setDiffuseTexture)
                .def("setRoughnessTexture", &atcg::DielectricMaterial::setRoughnessTexture)
                .def("setIorTexture", &atcg::DielectricMaterial::setIorTexture)
                .def("setDiffuseColor",
                     [](atcg::DielectricMaterial& material, const glm::vec3& color)
                     { material.setDiffuseColor(color); })
                .def("setDiffuseColor",
                     [](atcg::DielectricMaterial& material, const glm::vec4& color)
                     { material.setDiffuseColor(color); })
                .def("setRoughness", &atcg::DielectricMaterial::setRoughness)
                .def("setIor", &atcg::DielectricMaterial::setIor);

            m_null_material.def(py::init<>());

            m_material_registry
                .def(
                    "createMaterial",
                    [](const std::string& type, const atcg::Dictionary& dict) -> atcg::ref_ptr<atcg::Material>
                    {
                        auto registry = atcg::MaterialRegistry::getRegistry();
                        if(!registry->find(type))
                        {
                            ATCG_ERROR("Material type '{0}' is not registered", type);
                            return nullptr;
                        }
                        return atcg::MaterialRegistry::createMaterial(registry, type, dict);
                    },
                    "type"_a,
                    "dict"_a)
                .def(
                    "createMaterial",
                    [](const std::string& type) -> atcg::ref_ptr<atcg::Material>
                    {
                        auto registry = atcg::MaterialRegistry::getRegistry();
                        if(!registry->find(type))
                        {
                            ATCG_ERROR("Material type '{0}' is not registered", type);
                            return nullptr;
                        }
                        atcg::Dictionary dict;
                        return atcg::MaterialRegistry::createMaterial(registry, type, dict);
                    },
                    "type"_a)
                .def("getRegisteredMaterialTypes",
                     []() { return atcg::MaterialRegistry::getRegisteredMaterialTypes(); })
                .def(
                    "serializeMaterial",
                    [](const atcg::ref_ptr<atcg::Material>& material, const std::string& path)
                    {
                        if(!material)
                        {
                            ATCG_ERROR("Cannot serialize an empty material");
                            return;
                        }
                        auto registry = atcg::MaterialRegistry::getRegistry();
                        if(!registry->find(material->getMaterialType()))
                        {
                            ATCG_ERROR("Material type '{0}' is not registered", material->getMaterialType());
                            return;
                        }
                        atcg::MaterialRegistry::serializeMaterial(registry, material, std::filesystem::path(path));
                    },
                    "material"_a,
                    "path"_a);
#ifndef ATCG_HEADLESS
            m_material_registry.def(
                "renderMaterialGUI",
                [](const atcg::ref_ptr<atcg::Material>& material, const std::string& key) -> py::tuple
                {
                    bool deactivated = false;
                    if(!material)
                    {
                        ATCG_ERROR("Cannot render the GUI of an empty material");
                        return py::make_tuple(false, deactivated);
                    }
                    auto registry = atcg::MaterialRegistry::getRegistry();
                    if(!registry->find(material->getMaterialType()))
                    {
                        ATCG_ERROR("Material type '{0}' is not registered", material->getMaterialType());
                        return py::make_tuple(false, deactivated);
                    }
                    bool changed = atcg::MaterialRegistry::renderMaterialGUI(registry, material, key, deactivated);
                    return py::make_tuple(changed, deactivated);
                },
                "material"_a,
                "key"_a);
#endif
        });
}
}    // namespace python
}    // namespace atcg
