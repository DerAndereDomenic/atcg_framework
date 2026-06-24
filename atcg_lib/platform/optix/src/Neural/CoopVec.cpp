#include <Neural/CoopVec.h>

#include <Core/Common.h>

namespace atcg
{
void CoopVecMatrixConvert(OptixDeviceContext context,
                          CUstream stream,
                          unsigned int numNetworks,
                          const OptixNetworkDescription* inputNetworkDescription,
                          CUdeviceptr inputNetworks,
                          size_t inputNetworkStrideInBytes,
                          const OptixNetworkDescription* outputNetworkDescription,
                          CUdeviceptr outputNetworks,
                          size_t outputNetworkStrideInBytes)
{
    OPTIX_CHECK(optixCoopVecMatrixConvert(context,
                                          stream,
                                          numNetworks,
                                          inputNetworkDescription,
                                          inputNetworks,
                                          inputNetworkStrideInBytes,
                                          outputNetworkDescription,
                                          outputNetworks,
                                          outputNetworkStrideInBytes));
}

void CoopVecMatrixComputeSize(OptixDeviceContext context,
                              unsigned int N,
                              unsigned int K,
                              OptixCoopVecElemType elementType,
                              OptixCoopVecMatrixLayout layout,
                              size_t rowColumnStrideInBytes,
                              size_t* sizeInBytes)
{
    OPTIX_CHECK(optixCoopVecMatrixComputeSize(context, N, K, elementType, layout, rowColumnStrideInBytes, sizeInBytes));
}
}    // namespace atcg