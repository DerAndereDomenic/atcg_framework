#pragma once

#include <Core/RaytracingContext.h>
#include <Shape/ShapeInstance.h>

namespace atcg
{
class IAS
{
public:
    IAS(const atcg::ref_ptr<RaytracingContext>& context, const std::vector<atcg::ref_ptr<ShapeInstance>>& shapes);

    ATCG_INLINE OptixTraversableHandle getTraversableHandle() const { return _handle; }

private:
    OptixTraversableHandle _handle;
    atcg::DeviceBuffer<uint8_t> _ast_buffer;
};
}    // namespace atcg