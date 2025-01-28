#pragma once

#include <Core/Platform.h>
#include <Core/Memory.h>
#include <Core/SystemRegistry.h>
#include <Renderer/Context.h>

namespace atcg
{

/**
 * @brief A context manager system is used to manage contexts over several threads.
 */
class ContextManagerSystem
{
public:
    /**
     * @brief Default constructor
     */
    ContextManagerSystem();

    /**
     * @brief Destructor
     */
    ~ContextManagerSystem();

    /**
     * @brief Create the context
     * @note After creation this context will be the current context. Therefore, it is assumed that no context is
     * associated with this thread when this function is called.
     *
     * @param device_id The device id on which the context should be created
     * @note This is only used for headless rendering (on linux). For normal in-window rendering, this value is ignored
     *
     * @return The created context
     */
    atcg::ref_ptr<atcg::Context> createContext(const int device_id = 0);

    /**
     * @brief Create the context
     * This function is used to create a shared context.
     * @note After creation this context will be the current context. The device of this context will be the same as the
     * shared context.
     *
     * @param shared The context to share from
     *
     * @return The created context
     */
    atcg::ref_ptr<atcg::Context> createContext(const atcg::ref_ptr<atcg::Context>& shared);

    /**
     * @brief Destroy the context.
     * The context pointer will be set to null
     *
     * @param context The context to destroy
     */
    void destroyContext(atcg::ref_ptr<atcg::Context>& context);

    /**
     * @brief Get the current context for the calling thread
     *
     * @return The current context
     */
    atcg::ref_ptr<atcg::Context> getCurrentContext() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

namespace ContextManager
{
/**
 * @brief Create the context
 * @note After creation this context will be the current context. Therefore, it is assumed that no context is
 * associated with this thread when this function is called.
 *
 * @param device_id The device id on which the context should be created
 * @note This is only used for headless rendering (on linux). For normal in-window rendering, this value is ignored
 *
 * @return The created context
 */
ATCG_INLINE atcg::ref_ptr<atcg::Context> createContext(const int device_id = 0)
{
    return SystemRegistry::instance()->getSystem<atcg::ContextManagerSystem>()->createContext(device_id);
}

/**
 * @brief Create the context
 * This function is used to create a shared context.
 * @note After creation this context will be the current context. The device of this context will be the same as the
 * shared context.
 *
 * @param shared The context to share from
 *
 * @return The created context
 */
ATCG_INLINE atcg::ref_ptr<atcg::Context> createContext(const atcg::ref_ptr<atcg::Context>& shared)
{
    return SystemRegistry::instance()->getSystem<atcg::ContextManagerSystem>()->createContext(shared);
}

/**
 * @brief Destroy the context
 * The context pointer will be set to null
 *
 * @param context The context to destroy
 */
ATCG_INLINE void destroyContext(atcg::ref_ptr<atcg::Context>& context)
{
    SystemRegistry::instance()->getSystem<atcg::ContextManagerSystem>()->destroyContext(context);
}

/**
 * @brief Get the current context for the calling thread
 *
 * @return The current context
 */
ATCG_INLINE atcg::ref_ptr<atcg::Context> getCurrentContext()
{
    return SystemRegistry::instance()->getSystem<atcg::ContextManagerSystem>()->getCurrentContext();
}
}    // namespace ContextManager

}    // namespace atcg