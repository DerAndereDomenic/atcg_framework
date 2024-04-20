#pragma once

#include <DataStructure/AccelerationStructure.h>
#include <optix.h>

namespace atcg
{
class OptixAccelerationStructure : public AccelerationStructure
{
public:
    OptixAccelerationStructure(OptixDeviceContext context, const atcg::ref_ptr<Graph>& graph);

    ~OptixAccelerationStructure();

    OptixTraversableHandle getTraversableHandle() const { return _handle; }

private:
    atcg::DeviceBuffer<uint8_t> _gas_buffer;
    OptixTraversableHandle _handle;
};
}    // namespace atcg