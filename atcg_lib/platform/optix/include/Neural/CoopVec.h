#pragma once

#include <Core/API.h>
#include <Core/Platform.h>
#include <Core/Optix.h>

// Just wrappers for dll export of optix cooperative vector matrix conversion and size computation functions
namespace atcg
{
ATCG_API void CoopVecMatrixConvert(OptixDeviceContext context,
                                   CUstream stream,
                                   unsigned int numNetworks,
                                   const OptixNetworkDescription* inputNetworkDescription,
                                   CUdeviceptr inputNetworks,
                                   size_t inputNetworkStrideInBytes,
                                   const OptixNetworkDescription* outputNetworkDescription,
                                   CUdeviceptr outputNetworks,
                                   size_t outputNetworkStrideInBytes);

ATCG_API void CoopVecMatrixComputeSize(OptixDeviceContext context,
                                       unsigned int N,
                                       unsigned int K,
                                       OptixCoopVecElemType elementType,
                                       OptixCoopVecMatrixLayout layout,
                                       size_t rowColumnStrideInBytes,
                                       size_t* sizeInBytes);
}    // namespace atcg