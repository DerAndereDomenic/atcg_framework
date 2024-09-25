#pragma once

#include <Core/Memory.h>
#include <Renderer/Texture.h>
#include <DataStructure/JPEGConfig.h>

#include <vector>

#include <torch/types.h>

namespace atcg
{

/**
 * @brief A wrapper around nvjpeg for fast jpeg decompression.
 * @note This is only built with the CUDA backend.
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
     * @param flip_vertically If the image should be flipped vertically
     * @param backend The backend. default = SOFTWARE. Only pass HARDWARE if the GPU supports hardware decoding
     */
    JPEGDecoder(uint32_t num_images,
                uint32_t img_width,
                uint32_t img_height,
                bool flip_vertically = false,
                JPEGBackend backend  = JPEGBackend::SOFTWARE);

    /**
     * @brief Destructor
     */
    ~JPEGDecoder();

    /**
     * @brief Decompress a batch of images
     *
     * @param jpeg_files A vector containing the raw jpeg files
     *
     * @return The tensor containing the loaded images
     */
    torch::Tensor decompressImages(const std::vector<std::vector<uint8_t>>& jpeg_files);

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