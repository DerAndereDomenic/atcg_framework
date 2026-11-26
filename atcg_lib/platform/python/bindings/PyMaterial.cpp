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
                     [](const atcg::ref_ptr<atcg::Material>& self)
                     {
                         auto ptr = std::dynamic_pointer_cast<atcg::OpaqueMaterial>(self);
                         if(!ptr) throw std::runtime_error("Not an OpaqueMaterial");
                         return ptr;
                     })
                .def("asDielectric",
                     [](const atcg::ref_ptr<atcg::Material>& self)
                     {
                         auto ptr = std::dynamic_pointer_cast<atcg::DielectricMaterial>(self);
                         if(!ptr) throw std::runtime_error("Not a DielectricMaterial");
                         return ptr;
                     })
                .def("asNull",
                     [](const atcg::ref_ptr<atcg::Material>& self)
                     {
                         auto ptr = std::dynamic_pointer_cast<atcg::NullMaterial>(self);
                         if(!ptr) throw std::runtime_error("Not a NullMaterial");
                         return ptr;
                     })
                .def("getMaterialType", &atcg::Material::getMaterialType);

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
        });
}
}    // namespace python
}    // namespace atcg
