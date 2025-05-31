#include <Shape/IAS.h>

#include <Core/Common.h>

#include <optix_stubs.h>

namespace atcg
{
InstanceAccelerationStructure::InstanceAccelerationStructure(const atcg::ref_ptr<RaytracingContext>& context,
                                                             const std::vector<atcg::ref_ptr<ShapeInstance>>& shapes)
{
    std::vector<OptixInstance> optix_instances;
    int num_instances = 0;
    for(auto shape: shapes)
    {
        OptixInstance optix_instance = {0};
        memset(&optix_instance, 0, sizeof(OptixInstance));

        optix_instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        optix_instance.instanceId        = num_instances;
        optix_instance.sbtOffset         = num_instances;
        optix_instance.visibilityMask    = 1;
        optix_instance.traversableHandle = shape->getShape()->getAST();

        const glm::mat4& transform = shape->getTransform();

        reinterpret_cast<glm::mat3x4&>(optix_instance.transform) = glm::transpose(glm::mat4x3(transform));

        ++num_instances;

        optix_instances.push_back(optix_instance);
    }

    atcg::DeviceBuffer<OptixInstance> d_instances(num_instances);
    d_instances.upload(optix_instances.data());

    OptixBuildInput instance_input            = {};
    instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances    = (CUdeviceptr)d_instances.get();
    instance_input.instanceArray.numInstances = static_cast<uint32_t>(num_instances);

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context->getContextHandle(),
                                             &accel_options,
                                             &instance_input,
                                             1,    // num build inputs
                                             &ias_buffer_sizes));

    atcg::DeviceBuffer<uint8_t> d_temp_buffer(ias_buffer_sizes.tempSizeInBytes);
    _ast_buffer = atcg::DeviceBuffer<uint8_t>(ias_buffer_sizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(context->getContextHandle(),
                                nullptr,    // CUDA stream
                                &accel_options,
                                &instance_input,
                                1,    // num build inputs
                                (CUdeviceptr)d_temp_buffer.get(),
                                ias_buffer_sizes.tempSizeInBytes,
                                (CUdeviceptr)_ast_buffer.get(),
                                ias_buffer_sizes.outputSizeInBytes,
                                &_handle,
                                nullptr,    // emitted property list
                                0           // num emitted properties
                                ));
}
}    // namespace atcg