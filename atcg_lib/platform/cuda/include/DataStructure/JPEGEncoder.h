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
 * @note This is only built with the CUDA backend
 */
class JPEGEncoder
{
public:
    /**
     * @brief Constructor
     * @note The vertically flip option is true per default. This is due to compatibility reasons with opengl. If the
     * compressed to be compressed is an opengl texture and then saved to disk, the image will have the "correct"
     * orientation.
     *
     * @param flip_vertically If the image should be flipped vertically
     * @param backend The backend. default = SOFTWARE. Onyl pass HARDWARE if the GPU is capable of hardware encoding
     */
    JPEGEncoder(bool flip_vertically = true, JPEGBackend backend = JPEGBackend::SOFTWARE);

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