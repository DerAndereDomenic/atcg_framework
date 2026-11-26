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
    auto m_asset_manager = m.def_submodule("AssetManager");

    auto m_project = py::class_<atcg::Project, atcg::ref_ptr<atcg::Project>>(m, "Project");

    ctx.define(
        [=]() mutable
        {
            m_asset.def_readwrite("handle", &atcg::Asset::handle).def("getType", &atcg::Asset::getType);
            m_asset_type.value("None", atcg::AssetType::None)
                .value("Scene", atcg::AssetType::Scene)
                .value("Texture2D", atcg::AssetType::Texture2D)
                .value("Material", atcg::AssetType::Material)
                .value("Graph", atcg::AssetType::Graph)
                .value("Script", atcg::AssetType::Script)
                .value("Shader", atcg::AssetType::Shader);

            m_asset_data.def_readwrite("type", &atcg::AssetMetaData::type)
                .def_readwrite("name", &atcg::AssetMetaData::name);

            m_asset_manager_system.def(py::init<>())
                .def("getAsset", &atcg::AssetManagerSystem::getAsset)
                .def("getMetaData", &atcg::AssetManagerSystem::getMetaData)
                .def("isAssetHandleValid", &atcg::AssetManagerSystem::isAssetHandleValid)
                .def("isAssetLoaded", &atcg::AssetManagerSystem::isAssetLoaded)
                .def("updateName", &atcg::AssetManagerSystem::updateName)
                .def("registerAsset",
                     static_cast<atcg::AssetHandle (atcg::AssetManagerSystem::*)(const atcg::AssetMetaData& data)>(
                         &atcg::AssetManagerSystem::registerAsset))
                .def(
                    "registerAsset",
                    static_cast<atcg::AssetHandle (atcg::AssetManagerSystem::*)(const atcg::ref_ptr<atcg::Asset>& asset,
                                                                                const std::string& name,
                                                                                bool show_in_editor,
                                                                                bool serialize)>(
                        &atcg::AssetManagerSystem::registerAsset))
                .def("unloadAsset", &atcg::AssetManagerSystem::unloadAsset)
                .def("removeAsset", &atcg::AssetManagerSystem::removeAsset)
                .def("serializeRegistry", &atcg::AssetManagerSystem::serializeRegistry)
                .def("deserializeRegistry", &atcg::AssetManagerSystem::deserializeRegistry)
                .def("serializeAssets", &atcg::AssetManagerSystem::serializeAssets)
                .def("clear", &atcg::AssetManagerSystem::clear)
                .def("destroy", &atcg::AssetManagerSystem::destroy);

            m_asset_manager
                .def("getAsset", [](atcg::AssetHandle handle) { return atcg::AssetManager::getAsset(handle); })
                .def("getMetaData", &atcg::AssetManager::getMetaData)
                .def("isAssetHandleValid", &atcg::AssetManager::isAssetHandleValid)
                .def("isAssetLoaded", &atcg::AssetManager::isAssetLoaded)
                .def("updateName", &atcg::AssetManager::updateName)
                .def("registerAsset",
                     [](const atcg::AssetMetaData& data) { return atcg::AssetManager::registerAsset(data); })
                .def("registerAsset",
                     [](const atcg::ref_ptr<atcg::Asset>& asset,
                        const std::string& name,
                        bool show_in_editor,
                        bool serialize)
                     { return atcg::AssetManager::registerAsset(asset, name, show_in_editor, serialize); })
                .def("unloadAsset", &atcg::AssetManager::unloadAsset)
                .def("removeAsset", &atcg::AssetManager::removeAsset)
                .def("serializeRegistry", &atcg::AssetManager::serializeRegistry)
                .def("deserializeRegistry", &atcg::AssetManager::deserializeRegistry)
                .def("serializeAssets", &atcg::AssetManager::serializeAssets)
                .def("clear", &atcg::AssetManager::clear)
                .def("assetTypeToString", &atcg::assetTypeToString)
                .def("stringToAssetType", &atcg::stringToAssetType);

            m_project.def_static("create", &atcg::Project::create)
                .def_static("load", [](const std::string& path) { return atcg::Project::load(path); })
                .def("save", [](const atcg::ref_ptr<atcg::Project>& project) { return project->save(); })
                .def("save",
                     [](const atcg::ref_ptr<atcg::Project>& project, const std::string& path)
                     { return project->save(path); })
                .def("setActiveScene",
                     py::overload_cast<const atcg::ref_ptr<atcg::Scene>&>(&atcg::Project::setActiveScene))
                .def("setActiveScene", py::overload_cast<atcg::AssetHandle>(&atcg::Project::setActiveScene))
                .def("getActiveScene", &atcg::Project::getActiveScene)
                .def_static("getActive", &atcg::Project::getActive)
                .def_static("saveActive", []() { atcg::Project::saveActive(); })
                .def_static("saveActive", [](const std::string& path) { atcg::Project::saveActive(path); })
                .def("getFilePath",
                     [](const atcg::ref_ptr<atcg::Project>& project) { return project->getFilePath().string(); });
        });
}
}    // namespace python
}    // namespace atcg
