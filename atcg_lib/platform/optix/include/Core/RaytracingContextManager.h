#pragma once

#include <Core/SystemRegistry.h>
#include <Core/RaytracingContext.h>

namespace atcg
{
/**
 * @brief A system that handles ray tracing contexts
 */
class RaytracingContextManagerSystem
{
public:
    /**
     * @brief Constructor
     */
    RaytracingContextManagerSystem();

    /**
     * @brief Destructor
     */
    ~RaytracingContextManagerSystem();

    /**
     * @brief Create a context on a specific device
     * 
     * @param device_id The device id
     */
    atcg::ref_ptr<atcg::RaytracingContext> createContext(const int device_id = 0);

    /**
     * @brief Destroy a context
     * 
     * @param context The context
     */
    void destroyContext(atcg::ref_ptr<atcg::RaytracingContext>& context);

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

namespace RaytracingContextManager
{
/**
 * @brief Create a context on a specific device
 * 
 * @param device_id The device id
 */
ATCG_INLINE atcg::ref_ptr<atcg::RaytracingContext> createContext(const int device_id = 0)
{
    return SystemRegistry::instance()->getSystem<RaytracingContextManagerSystem>()->createContext(device_id);
}

/**
 * @brief Destroy a context
 * 
 * @param context The context
 */
ATCG_INLINE void destroyContext(atcg::ref_ptr<atcg::RaytracingContext>& context)
{
    SystemRegistry::instance()->getSystem<RaytracingContextManagerSystem>()->destroyContext(context);
}
}    // namespace RaytracingContextManager

}    // namespace atcg