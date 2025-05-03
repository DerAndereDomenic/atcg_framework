#pragma once

#include <Core/SystemRegistry.h>
#include <Core/RaytracingContext.h>

namespace atcg
{
class RaytracingContextManagerSystem
{
public:
    RaytracingContextManagerSystem();

    ~RaytracingContextManagerSystem();

    atcg::ref_ptr<atcg::RaytracingContext> createContext(const int device_id = 0);

    void destroyContext(atcg::ref_ptr<atcg::RaytracingContext>& context);

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

namespace RaytracingContextManager
{
ATCG_INLINE atcg::ref_ptr<atcg::RaytracingContext> createContext(const int device_id = 0)
{
    return SystemRegistry::instance()->getSystem<RaytracingContextManagerSystem>()->createContext(device_id);
}

ATCG_INLINE void destroyContext(atcg::ref_ptr<atcg::RaytracingContext>& context)
{
    SystemRegistry::instance()->getSystem<RaytracingContextManagerSystem>()->destroyContext(context);
}
}    // namespace RaytracingContextManager

}    // namespace atcg