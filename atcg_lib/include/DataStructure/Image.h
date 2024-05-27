#pragma once

#include <torch/types.h>

namespace atcg
{

/**
 * @brief A class to model an image.
 * This is mostly a convinient class for loading and storing images.
 * It does not support direct pixel/image manipulation other than gamma correction.
 * It should be passed to a texture for rendering porpurses.
 */
class Image
{
public:
    /**
     * @brief Default constructor.
     */
    Image() = default;

    /**
     * @brief Create an one byte per channel image from external data.
     * The data is copied.
     *
     * @param img_data The data
     * @param width The image width
     * @param height The image height
     * @param channels The number of channels between 1 and 4
     *
     */
    Image(uint8_t* data, uint32_t width, uint32_t height, uint32_t channels);

    /**
     * @brief Create a floating point image from external data.
     * The data is copied.
     *
     * @param img_data The data
     * @param width The image width
     * @param height The image height
     * @param channels The number of channels between 1 and 4
     *
     */
    Image(float* data, uint32_t width, uint32_t height, uint32_t channels);

    /**
     * @brief Create an image from tensor.
     *
     * @param tensor The image data of shape (h,w,c)
     * The type of the tensor should either be uint8 (for LDR) or float (for HDR)
     */
    Image(const torch::Tensor& tensor);

    /**
     * @brief Destructor
     */
    ~Image();

    /**
     * @brief Load an image.
     * All RGB images will be padded to RGBA. Single channel textures will remain single texture.
     * If the path has an .exr or .hdr ending it will be loaded as a HDR float image, otherwise as one byte per color
     * channel.
     *
     * @param filename The filename to store the image to
     */
    void load(const std::string& filename);

    /**
     * @brief Load an image.
     * If filename ends with .exr or .hdr it will be exported as an hdr float image (even if the underlying data is
     * supposed to be quantisized in bytes. No conversion is performed.)
     *
     * @param filename The filename to store the image to
     */
    void store(const std::string& filename);

    /**
     * @brief Apply gamma correction to the image
     *
     * @param gamma The gamma constant
     */
    void applyGamma(const float gamma);

    /**
     * @brief Set image data
     *
     * @param data The image data
     */
    void setData(uint8_t* data);

    /**
     * @brief Set image data
     *
     * @param data The image data
     */
    void setData(const torch::Tensor& data);

    /**
     * @brief Get the width of the image
     *
     * @return The width
     */
    ATCG_INLINE uint32_t width() const { return _width; }

    /**
     * @brief Get the height of the image
     *
     * @return The height
     */
    ATCG_INLINE uint32_t height() const { return _height; }

    /**
     * @brief Get the number of channels
     *
     * @return The number of channels
     */
    ATCG_INLINE uint32_t channels() const { return _channels; }

    /**
     * @brief Get the name/filepath of the image
     *
     * @return The filepath
     */
    ATCG_INLINE const std::string& name() const { return _filename; }

    /**
     * @brief If this image is a HDR texture
     *
     * @return True if it's an hdr image
     */
    ATCG_INLINE bool isHDR() const { return _hdr; }

    /**
     * @brief Get the image data interpreted in a specific format.
     *
     * @return The data
     */
    ATCG_INLINE torch::Tensor data() const { return _img_data; }

private:
    void loadLDR(const std::string& filename);
    void loadHDR(const std::string& filename);

    torch::Tensor _img_data;
    uint32_t _width    = 0;
    uint32_t _height   = 0;
    uint32_t _channels = 0;
    bool _hdr          = false;
    std::string _filename;
};

namespace IO
{
/**
 * @brief Read an image
 *
 * @param filename The path to the image
 * @param gamma The gamma correction constant
 *
 * @return The image.
 */
atcg::ref_ptr<Image> imread(const std::string& filename, const float gamma = 1.0f);

/**
 * @brief Store an image
 *
 * @param image The image to store
 * @param filename The file path
 * @param gamma The gamma correction constant
 *
 */
void imwrite(const atcg::ref_ptr<Image>& image, const std::string& filename, const float gamma = 1.0f);
}    // namespace IO

namespace Utils
{
/**
 * @brief Compute the absolute error maps between pairwise images.
 * The AE is defined as |x_i - y_i|. All images are expected to be in a floating point representation.
 *
 * @param ground_truth A batch with ground truth images of shape (B, H, W, C)
 * @param prediction A batch with predicted images of shape (B, H, W, C)
 * @param channel_reduction How the last dimension (channels C) is reduced. Either "none", "mean", "sum", or "norm"
 * which computes the L2 norm over the channel dimension. "none" computes a 3 channel error map.
 *
 * @return A (B, H, W, {C|1}) tensor with error maps. C=1 if a reduction other than "none" was used.
 */
torch::Tensor AEMap(const torch::Tensor& ground_truth,
                    const torch::Tensor& prediction,
                    const std::string& channel_reduction = "mean");

/**
 * @brief Compute the relative absolute error maps between pairwise images.
 * The relAE is defined as |x_i - y_i| / (|y_i| + delta). All images are expected to be in a floating point
 * representation.
 *
 * @param ground_truth A batch with ground truth images of shape (B, H, W, C)
 * @param prediction A batch with predicted images of shape (B, H, W, C)
 * @param channel_reduction How the last dimension (channels C) is reduced. Either "none", "mean", "sum", or "norm"
 * which computes the L2 norm over the channel dimension. "none" computes a 3 channel error map.
 * @param delta Delta value for numerical stability
 *
 * @return A (B, H, W, {C|1}) tensor with error maps. C=1 if a reduction other than "none" was used.
 */
torch::Tensor relAEMap(const torch::Tensor& ground_truth,
                       const torch::Tensor& prediction,
                       const std::string& channel_reduction = "mean",
                       const float delta                    = 1e-4f);

/**
 * @brief Compute the squared error maps between pairwise images.
 * The SE is defined as |x_i - y_i|^2. All images are expected to be in a floating point representation.
 *
 * @param ground_truth A batch with ground truth images of shape (B, H, W, C)
 * @param prediction A batch with predicted images of shape (B, H, W, C)
 * @param channel_reduction How the last dimension (channels C) is reduced. Either "none", "mean", "sum", or "norm"
 * which computes the L2 norm over the channel dimension. "none" computes a 3 channel error map.
 *
 * @return A (B, H, W, {C|1}) tensor with error maps. C=1 if a reduction other than "none" was used.
 */
torch::Tensor SEMap(const torch::Tensor& ground_truth,
                    const torch::Tensor& prediction,
                    const std::string& channel_reduction = "mean");

/**
 * @brief Compute the relative squared error maps between pairwise images.
 * The relAE is defined as |x_i - y_i|^2 / (|y_i|^2 + delta). All images are expected to be in a floating point
 * representation.
 *
 * @param ground_truth A batch with ground truth images of shape (B, H, W, C)
 * @param prediction A batch with predicted images of shape (B, H, W, C)
 * @param channel_reduction How the last dimension (channels C) is reduced. Either "none", "mean", "sum", or "norm"
 * which computes the L2 norm over the channel dimension. "none" computes a 3 channel error map.
 * @param delta Delta value for numerical stability
 *
 * @return A (B, H, W, {C|1}) tensor with error maps. C=1 if a reduction other than "none" was used.
 */
torch::Tensor relSEMap(const torch::Tensor& ground_truth,
                       const torch::Tensor& prediction,
                       const std::string& channel_reduction = "mean",
                       const float delta                    = 1e-4f);

/**
 * @brief Compute the pixelwise mean absolute error between pairwise images.
 * The MAE is defined as sum_i |x_i - y_i|. All images are expected to be in a floating point representation.
 *
 * @param ground_truth A batch with ground truth images of shape (B, H, W, C)
 * @param prediction A batch with predicted images of shape (B, H, W, C)
 * @param channel_reduction How the last dimension (channels C) is reduced. Either "none", "mean", "sum", or "norm"
 * which computes the L2 norm over the channel dimension. "none" computes a 3 channel error map.
 *
 * @return A (B, {C|1}) tensor with error values. C=1 if a reduction other than "none" was used.
 */
torch::Tensor
MAE(const torch::Tensor& ground_truth, const torch::Tensor& prediction, const std::string& channel_reduction = "mean");

/**
 * @brief Compute the pixelwise relative mean absolute error between pairwise images.
 * The relMAE is defined as sum_i |x_i - y_i| / (|x_i| + delta). All images are expected to be in a floating point
 * representation.
 *
 * @param ground_truth A batch with ground truth images of shape (B, H, W, C)
 * @param prediction A batch with predicted images of shape (B, H, W, C)
 * @param channel_reduction How the last dimension (channels C) is reduced. Either "none", "mean", "sum", or "norm"
 * which computes the L2 norm over the channel dimension. "none" computes a 3 channel error map.
 * @param delta Delta value for numerical stability
 *
 * @return A (B, {C|1}) tensor with error values. C=1 if a reduction other than "none" was used.
 */
torch::Tensor relMAE(const torch::Tensor& ground_truth,
                     const torch::Tensor& prediction,
                     const std::string& channel_reduction = "mean",
                     const float delta                    = 1e-4f);
/**
 * @brief Compute the pixelwise mean squared error between pairwise images.
 * The MSE is defined as sum_i |x_i - y_i|^2. All images are expected to be in a floating point representation.
 *
 * @param ground_truth A batch with ground truth images of shape (B, H, W, C)
 * @param prediction A batch with predicted images of shape (B, H, W, C)
 * @param channel_reduction How the last dimension (channels C) is reduced. Either "none", "mean", "sum", or "norm"
 * which computes the L2 norm over the channel dimension. "none" computes a 3 channel error map.
 *
 * @return A (B, {C|1}) tensor with error values. C=1 if a reduction other than "none" was used.
 */
torch::Tensor
MSE(const torch::Tensor& ground_truth, const torch::Tensor& prediction, const std::string& channel_reduction = "mean");

/**
 * @brief Compute the pixelwise relative mean squared error between pairwise images.
 * The relMSE is defined as sum_i |x_i - y_i|^2 / (|x_i|^2 + delta). All images are expected to be in a floating point
 * representation.
 *
 * @param ground_truth A batch with ground truth images of shape (B, H, W, C)
 * @param prediction A batch with predicted images of shape (B, H, W, C)
 * @param channel_reduction How the last dimension (channels C) is reduced. Either "none", "mean", "sum", or "norm"
 * which computes the L2 norm over the channel dimension. "none" computes a 3 channel error map.
 * @param delta Delta value for numerical stability
 *
 * @return A (B, {C|1}) tensor with error values. C=1 if a reduction other than "none" was used.
 */
torch::Tensor relMSE(const torch::Tensor& ground_truth,
                     const torch::Tensor& prediction,
                     const std::string& channel_reduction = "mean",
                     const float delta                    = 1e-4f);

}    // namespace Utils

}    // namespace atcg