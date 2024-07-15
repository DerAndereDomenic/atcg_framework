#pragma once

#include <Core/Memory.h>
#include <Renderer/Texture.h>
#include <vector>

#include <torch/types.h>

namespace atcg
{

/**
 * @brief A wrapper around nvjpeg for fast jpeg decompression.
 * When built with CUDA (ATCG_CUDA_BACKEND=true), this class will use nvjpeg and return device tensors. Otherwise, a CPU
 * loader will be used and will return host tensors.
 * TODO
 */
class JPEGDecoder
{
public:
    /**
     * @brief Default constructor
     */
    JPEGDecoder() = default;

    /**
     * @brief Constructor
     *
     * @param num_images The number of images
     * @param img_width The width of each image
     * @param img_height The height of each image
     */
    JPEGDecoder(uint32_t num_images, uint32_t img_width, uint32_t img_height);

    /**
     * @brief Destructor
     */
    ~JPEGDecoder();

    /**
     * @brief Decompress a batch of images
     *
     * @param filenames A vector containing filenames of images to be decompressed
     *
     * @return The tensor containing the loaded images
     */
    torch::Tensor decompressImages(const std::vector<std::string>& filenames);

    /**
     * @brief Decompress a batch of images
     *
     * @param filenames A vector containing filenames of images to be decompressed
     * @param valid A tensor of valid cameras
     *
     * @return The tensor containing the loaded images
     */
    torch::Tensor decompressImages(const std::vector<std::string>& filenames, const torch::Tensor& valid);

    /**
     * @brief Upload and return the decompressed image tensor to the renderer
     * The texture should be a (num_images, img_width, img_height, 4) sized texture of type uint8.
     *
     * @param texture The texture
     */
    void copyToOutput(const atcg::ref_ptr<Texture3D>& texture);

    /**
     * @brief Upload and return the decompressed image tensor to the renderer
     * The texture should be a (num_images, img_width, img_height, 4) sized texture of type uint8.
     *
     * @param texture The texture
     */
    void copyToOutput(atcg::textureArray texture);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg