#pragma once

#include <Shape/ShapeInstance.h>

namespace atcg
{
class IAS
{
public:
    IAS(OptixDeviceContext context, const std::vector<atcg::ref_ptr<ShapeInstance>>& shapes);

    ATCG_INLINE OptixTraversableHandle getTraversableHandle() const { return _handle; }

private:
    OptixTraversableHandle _handle;
    atcg::DeviceBuffer<uint8_t> _ast_buffer;
};
}    // namespace atcg