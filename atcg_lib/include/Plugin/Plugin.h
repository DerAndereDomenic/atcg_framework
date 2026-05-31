#pragma once

#include <Core/Memory.h>
#include <Core/Application.h>
#include <Core/SystemRegistry.h>
#include <Scene/Scene.h>
#include <Renderer/Camera.h>
#include <Renderer/GraphicsAPI.h>
#include <imgui.h>

#include <filesystem>
#include <unordered_map>
#include <functional>

namespace atcg
{
class PluginManager
{
public:
    using SharedLibraryHandle = void*;

    ~PluginManager();

    bool loadPlugin(const std::filesystem::path& path);

    bool releasePlugin(const std::filesystem::path& path);

    template<typename BaseT, typename... Args>
    std::invoke_result_t<typename BaseT::PluginCreate, Args...> createClass(std::string_view type, Args&&... args) const
    {
        auto it = _registered_classes.find(type.data());
        if(it != _registered_classes.end())
        {
            auto* desc = dynamic_cast<ClassDesc<BaseT>*>(it->second.get());
            if(desc)
            {
                return desc->create(std::forward<Args>(args)...);
            }
            else
            {
                // Handle error: class type found but base type mismatch
                return std::invoke_result_t<typename BaseT::PluginCreate, Args...> {nullptr};
            }
        }
        else
        {
            // Handle error: class type not found
            return std::invoke_result_t<typename BaseT::PluginCreate, Args...> {nullptr};
        }
    }

private:
    struct ClassDescBase
    {
        ClassDescBase(SharedLibraryHandle handle, std::string_view type) : library_handle(handle), type(type) {}
        virtual ~ClassDescBase() {}

        SharedLibraryHandle library_handle;
        std::string type;
    };

    template<typename BaseT>
    struct ClassDesc : public ClassDescBase
    {
        using PluginCreate = typename BaseT::PluginCreate;

        ClassDesc(SharedLibraryHandle handle, std::string_view type, PluginCreate create)
            : ClassDescBase(handle, type),
              create(create)
        {
        }

        PluginCreate create;
    };

    template<typename BaseT>
    void registerClass(SharedLibraryHandle handle, std::string_view type, typename BaseT::PluginCreate create)
    {
        auto it = _registered_classes.find(type.data());
        if(it != _registered_classes.end())
        {
            // Handle error: class type already registered
            return;
        }

        auto desc                        = std::make_unique<ClassDesc<BaseT>>(handle, type, create);
        _registered_classes[type.data()] = std::move(desc);
    }

private:
    std::unordered_map<std::filesystem::path, SharedLibraryHandle> _loaded_plugins;
    std::unordered_map<std::string, std::unique_ptr<ClassDescBase>> _registered_classes;


    friend class PluginRegistry;
};

class PluginRegistry
{
public:
    PluginRegistry(PluginManager& manager, PluginManager::SharedLibraryHandle handle)
        : _manager(manager),
          _handle(handle)
    {
    }

    PluginRegistry(const PluginRegistry&)            = delete;
    PluginRegistry& operator=(const PluginRegistry&) = delete;

    template<typename BaseT>
    void registerClass(std::string_view type, typename BaseT::PluginCreate create)
    {
        _manager.registerClass<BaseT>(_handle, type, create);
    }

    template<typename BaseT, typename DerivedT>
    void registerClass(std::string_view type)
    {
        registerClass<BaseT>(T::pluginType, T::create);
    }

private:
    PluginManager& _manager;
    PluginManager::SharedLibraryHandle _handle;
};

#define ATCG_PLUGIN_BASE_CLASS(base_class)                                                                             \
public:                                                                                                                \
    static const std::string& getBaseClassName()                                                                       \
    {                                                                                                                  \
        static std::string type(#base_class);                                                                          \
        return type;                                                                                                   \
    }                                                                                                                  \
    virtual const std::string& getPluginType() const = 0;

#define ATCG_PLUGIN_CLASS(class)                                                                                       \
public:                                                                                                                \
    static inline const std::string pluginType = #class;                                                               \
    virtual const std::string& getPluginType() const final                                                             \
    {                                                                                                                  \
        return pluginType;                                                                                             \
    }

#define ATCG_PLUGIN_LIBRARY()                                                                                          \
    extern "C" __declspec(dllexport) void registerSystems(atcg::Application* app,                                      \
                                                          atcg::SystemRegistry* registry,                              \
                                                          ImGuiContext* imgui_context)                                 \
    {                                                                                                                  \
        atcg::Application::setApplicationInstance(app);                                                                \
        atcg::SystemRegistry::setInstance(registry);                                                                   \
        ImGui::SetCurrentContext(imgui_context);                                                                       \
        registry->getSystem<atcg::GraphicsAPI>()->init();                                                              \
    }

class DummyPluginBase
{
public:
    using PluginCreate = std::function<std::shared_ptr<DummyPluginBase>()>;
    ATCG_PLUGIN_BASE_CLASS(DummyPluginBase);

    virtual void doStuff(const atcg::ref_ptr<atcg::Scene>& scene, const atcg::ref_ptr<atcg::Camera>& camera) = 0;
};

}    // namespace atcg