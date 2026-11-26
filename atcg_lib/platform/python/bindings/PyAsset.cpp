#include "Bindings.h"

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
// Assets, asset manager and projects
void bindAsset(BindingContext& ctx)
{
    py::module_ m = ctx.module();

    auto m_asset = py::class_<atcg::Asset, atcg::ref_ptr<atcg::Asset>>(m, "Asset");

    auto m_assethandle = py::class_<atcg::AssetHandle>(m, "AssetHandle");
    auto m_asset_type  = py::enum_<atcg::AssetType>(m, "AssetType");
    auto m_asset_data  = py::class_<atcg::AssetMetaData>(m, "AssetMetaData");
    auto m_asset_manager_system =
        py::class_<atcg::AssetManagerSystem, atcg::ref_ptr<atcg::AssetManagerSystem>>(m, "AssetManagerSystem");
    auto m_asset_manager  = m.def_submodule("AssetManager");
    auto m_asset_importer = m.def_submodule("AssetImporter");
    auto m_asset_exporter = m.def_submodule("AssetExporter");

    auto m_project = py::class_<atcg::Project, atcg::ref_ptr<atcg::Project>>(m, "Project");

    ctx.define(
        [=]() mutable
        {
            m_asset.def_readwrite("handle", &atcg::Asset::handle).def("getType", &atcg::Asset::getType);
            m_asset_type.value("None", atcg::AssetType::None)
                .value("Scene", atcg::AssetType::Scene)
                .value("Texture2D", atcg::AssetType::Texture2D)
                .value("Texture3D", atcg::AssetType::Texture3D)
                .value("Material", atcg::AssetType::Material)
                .value("Graph", atcg::AssetType::Graph)
                .value("Script", atcg::AssetType::Script)
                .value("Shader", atcg::AssetType::Shader);

            m_assethandle.def(py::init<>())
                .def(py::init<uint64_t>(), "uuid"_a)
                .def("__int__", [](const atcg::AssetHandle& handle) { return (uint64_t)handle; })
                .def(
                    "__eq__",
                    [](const atcg::AssetHandle& a, const atcg::AssetHandle& b) { return (uint64_t)a == (uint64_t)b; },
                    py::is_operator())
                .def(
                    "__ne__",
                    [](const atcg::AssetHandle& a, const atcg::AssetHandle& b) { return (uint64_t)a != (uint64_t)b; },
                    py::is_operator())
                .def("__hash__", [](const atcg::AssetHandle& handle) { return py::hash(py::int_((uint64_t)handle)); })
                .def("__repr__",
                     [](const atcg::AssetHandle& handle)
                     { return "AssetHandle(" + std::to_string((uint64_t)handle) + ")"; });
            py::implicitly_convertible<uint64_t, atcg::AssetHandle>();

            m_asset_data.def(py::init<>())
                .def_readwrite("type", &atcg::AssetMetaData::type)
                .def_readwrite("name", &atcg::AssetMetaData::name)
                .def_readwrite("show_in_editor", &atcg::AssetMetaData::show_in_editor)
                .def_readwrite("serialize", &atcg::AssetMetaData::serialize)
                .def("__bool__", [](const atcg::AssetMetaData& data) { return (bool)data; });

            m_asset_manager_system.def(py::init<>())
                .def("getAsset", &atcg::AssetManagerSystem::getAsset, "handle"_a)
                .def("getMetaData", &atcg::AssetManagerSystem::getMetaData, "handle"_a)
                .def("getAssetRegistry", &atcg::AssetManagerSystem::getAssetRegistry)
                .def("isAssetHandleValid", &atcg::AssetManagerSystem::isAssetHandleValid, "handle"_a)
                .def("isAssetLoaded", &atcg::AssetManagerSystem::isAssetLoaded, "handle"_a)
                .def("updateName", &atcg::AssetManagerSystem::updateName, "handle"_a, "name"_a)
                .def("registerAsset",
                     static_cast<atcg::AssetHandle (atcg::AssetManagerSystem::*)(const atcg::AssetMetaData& data)>(
                         &atcg::AssetManagerSystem::registerAsset),
                     "data"_a)
                .def("registerAsset",
                     static_cast<atcg::AssetHandle (atcg::AssetManagerSystem::*)(atcg::AssetHandle handle,
                                                                                 const atcg::AssetMetaData& data)>(
                         &atcg::AssetManagerSystem::registerAsset),
                     "handle"_a,
                     "data"_a)
                .def(
                    "registerAsset",
                    static_cast<atcg::AssetHandle (atcg::AssetManagerSystem::*)(const atcg::ref_ptr<atcg::Asset>& asset,
                                                                                const std::string& name,
                                                                                bool show_in_editor,
                                                                                bool serialize)>(
                        &atcg::AssetManagerSystem::registerAsset),
                    "asset"_a,
                    "name"_a,
                    "show_in_editor"_a = true,
                    "serialize"_a      = true)
                .def("unloadAsset", &atcg::AssetManagerSystem::unloadAsset, "handle"_a)
                .def("removeAsset", &atcg::AssetManagerSystem::removeAsset, "handle"_a)
                .def(
                    "serializeRegistry",
                    [](atcg::AssetManagerSystem& system, const std::string& registry_path)
                    { system.serializeRegistry(registry_path); },
                    "registry_path"_a)
                .def(
                    "deserializeRegistry",
                    [](atcg::AssetManagerSystem& system, const std::string& registry_path)
                    { system.deserializeRegistry(registry_path); },
                    "registry_path"_a)
                .def(
                    "serializeAssets",
                    [](atcg::AssetManagerSystem& system, const std::string& root_path)
                    { system.serializeAssets(root_path); },
                    "root_path"_a)
                .def("clear", &atcg::AssetManagerSystem::clear)
                .def("destroy", &atcg::AssetManagerSystem::destroy)
                .def("loadStandardAssets", &atcg::AssetManagerSystem::loadStandardAssets)
                .def("registerStandardAssets", &atcg::AssetManagerSystem::registerStandardAssets)
                .def("getSphereMesh", &atcg::AssetManagerSystem::getSphereMesh)
                .def("getCylinderMesh", &atcg::AssetManagerSystem::getCylinderMesh)
                .def("getLUTTexture", &atcg::AssetManagerSystem::getLUTTexture)
                .def("getCameraFrustumMesh", &atcg::AssetManagerSystem::getCameraFrustumMesh)
                .def("getQuadMesh", &atcg::AssetManagerSystem::getQuadMesh)
                .def("getCubeMesh", &atcg::AssetManagerSystem::getCubeMesh)
                .def("getDummySkybox", &atcg::AssetManagerSystem::getDummySkybox)
                .def("getDefaultMaterial", &atcg::AssetManagerSystem::getDefaultMaterial);

            m_asset_manager
                .def(
                    "getAsset",
                    [](atcg::AssetHandle handle) { return atcg::AssetManager::getAsset(handle); },
                    "handle"_a)
                .def("getMetaData", &atcg::AssetManager::getMetaData, "handle"_a)
                .def("getAssetRegistry", &atcg::AssetManager::getAssetRegistry)
                .def("isAssetHandleValid", &atcg::AssetManager::isAssetHandleValid, "handle"_a)
                .def("isAssetLoaded", &atcg::AssetManager::isAssetLoaded, "handle"_a)
                .def("updateName", &atcg::AssetManager::updateName, "handle"_a, "name"_a)
                .def(
                    "registerAsset",
                    [](const atcg::AssetMetaData& data) { return atcg::AssetManager::registerAsset(data); },
                    "data"_a)
                .def(
                    "registerAsset",
                    [](atcg::AssetHandle handle, const atcg::AssetMetaData& data)
                    { return atcg::AssetManager::registerAsset(handle, data); },
                    "handle"_a,
                    "data"_a)
                .def(
                    "registerAsset",
                    [](const atcg::ref_ptr<atcg::Asset>& asset,
                       const std::string& name,
                       bool show_in_editor,
                       bool serialize)
                    { return atcg::AssetManager::registerAsset(asset, name, show_in_editor, serialize); },
                    "asset"_a,
                    "name"_a,
                    "show_in_editor"_a = true,
                    "serialize"_a      = true)
                .def("unloadAsset", &atcg::AssetManager::unloadAsset, "handle"_a)
                .def("removeAsset", &atcg::AssetManager::removeAsset, "handle"_a)
                .def(
                    "serializeRegistry",
                    [](const std::string& registry_path) { atcg::AssetManager::serializeRegistry(registry_path); },
                    "registry_path"_a)
                .def(
                    "deserializeRegistry",
                    [](const std::string& registry_path) { atcg::AssetManager::deserializeRegistry(registry_path); },
                    "registry_path"_a)
                .def(
                    "serializeAssets",
                    [](const std::string& root_path) { atcg::AssetManager::serializeAssets(root_path); },
                    "root_path"_a)
                .def("clear", &atcg::AssetManager::clear)
                .def("loadStandardAssets", &atcg::AssetManager::loadStandardAssets)
                .def("getSphereMesh", &atcg::AssetManager::getSphereMesh)
                .def("getCylinderMesh", &atcg::AssetManager::getCylinderMesh)
                .def("getLUTTexture", &atcg::AssetManager::getLUTTexture)
                .def("getCameraFrustumMesh", &atcg::AssetManager::getCameraFrustumMesh)
                .def("getQuadMesh", &atcg::AssetManager::getQuadMesh)
                .def("getCubeMesh", &atcg::AssetManager::getCubeMesh)
                .def("getDummySkybox", &atcg::AssetManager::getDummySkybox)
                .def("getDefaultMaterial", &atcg::AssetManager::getDefaultMaterial)
                .def("assetTypeToString", &atcg::assetTypeToString, "type"_a)
                .def("stringToAssetType", &atcg::stringToAssetType, "str"_a);

            m_asset_importer.def(
                "importAsset",
                [](const std::string& path, atcg::AssetHandle handle, const atcg::AssetMetaData& metadata)
                { return atcg::AssetImporter::importAsset(path, handle, metadata); },
                "path"_a,
                "handle"_a,
                "metadata"_a);

            m_asset_exporter.def(
                "exportAsset",
                [](const std::string& path, const atcg::ref_ptr<atcg::Asset>& asset, const atcg::AssetMetaData& data)
                { atcg::AssetExporter::exportAsset(path, asset, data); },
                "path"_a,
                "asset"_a,
                "data"_a);

            m_project
                .def_static(
                    "create",
                    [](const std::string& path) { return atcg::Project::create(path); },
                    "path"_a)
                .def_static(
                    "load",
                    [](const std::string& path) { return atcg::Project::load(path); },
                    "path"_a)
                .def("save", [](const atcg::ref_ptr<atcg::Project>& project) { return project->save(); })
                .def(
                    "save",
                    [](const atcg::ref_ptr<atcg::Project>& project, const std::string& path)
                    { return project->save(path); },
                    "path"_a)
                .def("setActiveScene",
                     py::overload_cast<const atcg::ref_ptr<atcg::Scene>&>(&atcg::Project::setActiveScene),
                     "scene"_a)
                .def("setActiveScene", py::overload_cast<atcg::AssetHandle>(&atcg::Project::setActiveScene), "handle"_a)
                .def("getActiveScene", &atcg::Project::getActiveScene)
#ifdef ATCG_CUDA_BACKEND
                .def("getIntegratorSettings", &atcg::Project::getIntegratorSettings)
                .def("setIntegratorSettings", &atcg::Project::setIntegratorSettings, "settings"_a)
#endif
                .def_static("getActive", &atcg::Project::getActive)
                .def_static("saveActive", []() { atcg::Project::saveActive(); })
                .def_static(
                    "saveActive",
                    [](const std::string& path) { atcg::Project::saveActive(path); },
                    "path"_a)
                .def("getFilePath",
                     [](const atcg::ref_ptr<atcg::Project>& project) { return project->getFilePath().string(); });
        });
}
}    // namespace python
}    // namespace atcg
