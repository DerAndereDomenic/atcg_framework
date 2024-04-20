#pragma once

#define OPTIX_CHECK(call)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        OptixResult res = call;                                                                                        \
        if(res != OPTIX_SUCCESS)                                                                                       \
        {                                                                                                              \
            std::stringstream ss;                                                                                      \
            ss << "Optix error '" << optixGetErrorName(res) << "' at " << __FILE__ << ":" << __LINE__ << ": "          \
               << #call;                                                                                               \
            ATCG_ERROR(ss.str());                                                                                      \
        }                                                                                                              \
    } while(0)

namespace detail
{

inline void convertToTextureObject(const torch::Tensor& texture_data,
                                   cudaArray_t& output_array,
                                   cudaTextureObject_t& output_texture)
{
    torch::Tensor data = texture_data;
    if(texture_data.size(2) == 2)
    {
        data = torch::cat({texture_data, torch::zeros_like(texture_data)}, -1);
    }

    if(texture_data.size(2) == 3)
    {
        data =
            torch::cat({texture_data,
                        torch::zeros_like(
                            texture_data.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}).unsqueeze(-1))},
                       -1);
    }


    uint32_t num_channels = data.size(2);
    bool isFloat          = data.dtype() == torch::kFloat32;

    uint32_t component_size = isFloat ? 32 : 8;

    cudaChannelFormatDesc format = {};
    format.f                     = isFloat ? cudaChannelFormatKindFloat : cudaChannelFormatKindUnsigned;
    format.x                     = component_size;
    format.y                     = num_channels == 4 ? component_size : 0;
    format.z                     = num_channels == 4 ? component_size : 0;
    format.w                     = num_channels == 4 ? component_size : 0;

    cudaExtent extent = {};
    extent.width      = data.size(1);
    extent.height     = data.size(0);
    extent.depth      = 0;

    CUDA_SAFE_CALL(cudaMalloc3DArray(&output_array, &format, extent));

    CUDA_SAFE_CALL(cudaMemcpy2DToArray(output_array,
                                       0,
                                       0,
                                       data.contiguous().data_ptr(),
                                       data.size(1) * data.size(2) * data.element_size(),
                                       data.size(1) * data.size(2) * data.element_size(),
                                       data.size(0),
                                       cudaMemcpyDeviceToDevice));

    cudaTextureDesc desc  = {};
    desc.normalizedCoords = 1;
    desc.addressMode[0]   = cudaAddressModeClamp;
    desc.addressMode[1]   = cudaAddressModeClamp;
    desc.addressMode[2]   = cudaAddressModeClamp;
    desc.readMode         = isFloat ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
    desc.filterMode       = cudaFilterModeLinear;

    cudaResourceDesc resDesc = {};
    resDesc.resType          = cudaResourceTypeArray;
    resDesc.res.array.array  = output_array;

    CUDA_SAFE_CALL(cudaCreateTextureObject(&output_texture, &resDesc, &desc, nullptr));
}
}    // namespace detail