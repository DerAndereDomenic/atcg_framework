#pragma once

#include <Core/CUDA.h>

namespace detail
{
inline void
convertToTextureObject(const torch::Tensor& tensor, torch::Tensor& output_texture, atcg::textureArray& output_array)
{
    output_texture        = tensor;
    output_array.width    = tensor.size(1);
    output_array.height   = tensor.size(0);
    output_array.channels = tensor.size(2);
    output_array.is_hdr   = tensor.scalar_type() == torch::kFloat32;
    output_array.data     = tensor.data_ptr();
}
}    // namespace detail