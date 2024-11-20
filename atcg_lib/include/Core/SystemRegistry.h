#pragma once

namespace atcg
{

/**
 * @brief A (singleton) class that handles different systems of the engine, i.e., Renderer, ShaderManager, Logger,
 * Input, etc. This class does not manage the lifetime of the registered systems but only grants global access to the
 * systems.
 */
class SystemRegistry
{
public:
    SystemRegistry(SystemRegistry const&) = delete;

    SystemRegistry& operator=(SystemRegistry const&) = delete;

    /**
     * @brief Initialize the registry. This is done by the application.
     */
    static void init();

    /**
     * @brief Get the instance.
     *
     * @return The instance
     */
    static SystemRegistry* instance();

    /**
     * @brief Initialize the registry by an outside source, for example to initialize dynamically loaded dlls.
     *
     * @param regsitry The registry
     */
    static void setInstance(SystemRegistry* registry);

    /**
     * @brief Shutdown the system. This is done by the application.
     */
    static void shutdown();

    /**
     * @brief Initialize the system
     *
     * @tparam T The data type
     * @param system The system to register
     */
    template<typename T>
    void registerSystem(T* system)
    {
        _systems[typeid(T).name()] = system;
    }

    /**
     * @brief Unregister a system
     *
     * @tparam T The data type
     */
    template<typename T>
    void unregisterSystem()
    {
        _systems.erase(typeid(T).name());
    }

    /**
     * @brief Check if a System is registered
     *
     * @tparam T The parameter
     * @return True if the system has a specific system
     */
    template<typename T>
    bool hasSystem() const
    {
        return _systems.find(typeid(T).name()) != _systems.end();
    }

    /**
     * @brief Get a system
     *
     * @tparam T The data type
     *
     * @return The system
     */
    template<typename T>
    T* getSystem()
    {
        return static_cast<T*>(_systems[typeid(T).name()]);
    }

private:
    SystemRegistry() = default;

    ~SystemRegistry() = default;

    std::unordered_map<std::string, void*> _systems;
};
}    // namespace atcg