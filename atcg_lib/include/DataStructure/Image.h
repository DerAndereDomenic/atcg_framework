#pragma once

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
    void setData(const uint8_t* data);

    /**
     * @brief Get the width of the image
     *
     * @return The width
     */
    inline uint32_t width() const { return _width; }

    /**
     * @brief Get the height of the image
     *
     * @return The height
     */
    inline uint32_t height() const { return _height; }

    /**
     * @brief Get the number of channels
     *
     * @return The number of channels
     */
    inline uint32_t channels() const { return _channels; }

    /**
     * @brief Get the name/filepath of the image
     *
     * @return The filepath
     */
    inline const std::string& name() const { return _filename; }

    /**
     *  @brief Get the raw data pointer of the image.
     *
     * @return The data
     */
    inline const uint8_t* data() const { return _img_data; }

    /**
     * @brief Get the image data interpreted in a specific format.
     *
     * @tparam T The type
     *
     * @return The data
     */
    template<typename T>
    inline const T* data() const
    {
        return reinterpret_cast<T*>(_img_data);
    }

private:
    void loadLDR(const std::string& filename);
    void loadHDR(const std::string& filename);

    uint8_t* _img_data;
    uint32_t _width    = 0;
    uint32_t _height   = 0;
    uint32_t _channels = 0;
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
}    // namespace atcg