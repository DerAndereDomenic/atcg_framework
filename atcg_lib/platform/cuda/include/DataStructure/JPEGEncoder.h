#pragma once

#include <Core/Memory.h>
#include <Renderer/Texture.h>
#include <DataStructure/JPEGConfig.h>

#include <vector>

#include <torch/types.h>

namespace atcg
{

/**
 * @brief A wrapper around nvjpeg for fast jpeg compression.
 * When built with CUDA (ATCG_CUDA_BACKEND=true), this class will use nvjpeg and return device tensors.
 * @note Currently, there is no CPU implementation.
 */
class JPEGEncoder
{
public:
    /**
     * @brief Constructor
     *
     * @param backend The backend. default = SOFTWARE. Onyl pass HARDWARE if the GPU is capable of hardware encoding
     */
    JPEGEncoder(JPEGBackend backend = JPEGBackend::SOFTWARE);

    /**
     * @brief Destructor
     */
    ~JPEGEncoder();

    /**
     * @brief Encode an image.
     *
     * @param img The image to encode
     * @return The compressed image
     */
    torch::Tensor compress(const torch::Tensor& img);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg