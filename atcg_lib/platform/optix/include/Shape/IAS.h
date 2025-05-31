#pragma once

#include <Core/RaytracingContext.h>
#include <Shape/ShapeInstance.h>

namespace atcg
{
/**
 * @brief A class to model an acceleration structure for multiple instances
 */
class InstanceAccelerationStructure
{
public:
    /**
     * @brief Constructor
     *
     * @param context The raytracing context
     * @param shapes The vector of shapes to build the IAS over
     */
    InstanceAccelerationStructure(const atcg::ref_ptr<RaytracingContext>& context,
                                  const std::vector<atcg::ref_ptr<ShapeInstance>>& shapes);

    /**
     * @brief Get the traversable handle
     *
     * @return The handle
     */
    ATCG_INLINE OptixTraversableHandle getTraversableHandle() const { return _handle; }

private:
    OptixTraversableHandle _handle;
    atcg::DeviceBuffer<uint8_t> _ast_buffer;
};
}    // namespace atcg