#pragma once

#include <Core/Memory.h>
#include <Core/Platform.h>

#include <optix.h>

namespace atcg
{
class RaytracingContextManagerSystem;

class RaytracingContext
{
public:
    void initRaytracingAPI();

    ATCG_INLINE OptixDeviceContext getContextHandle() const { return _context; }

private:
    RaytracingContext() = default;

    void destroy();

    void create(const int device_id = 0);

    RaytracingContext(const RaytracingContext&)            = delete;
    RaytracingContext& operator=(const RaytracingContext&) = delete;

private:
    OptixDeviceContext _context = nullptr;

    friend class RaytracingContextManagerSystem;
};
}    // namespace atcg