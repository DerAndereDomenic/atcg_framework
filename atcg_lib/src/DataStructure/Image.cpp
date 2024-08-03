#include <DataStructure/Image.h>

#include <stb_image.h>
#include <stb_image_write.h>

#include <Renderer/Texture.h>
#include <DataStructure/TorchUtils.h>

namespace atcg
{

Image::Image(uint8_t* data, uint32_t width, uint32_t height, uint32_t channels)
{
    setData(data, width, height, channels);
}

Image::Image(float* data, uint32_t width, uint32_t height, uint32_t channels)
{
    setData(data, width, height, channels);
}

Image::Image(const torch::Tensor& tensor)
{
    setData(tensor);
}

Image::Image(const atcg::ref_ptr<Texture2D>& texture)
{
    setData(texture->getData(atcg::CPU));
}

Image::~Image()
{
    uint32_t _width    = 0;
    uint32_t _height   = 0;
    uint32_t _channels = 0;
    _hdr               = false;
}

void Image::load(const std::string& filename)
{
    stbi_set_flip_vertically_on_load(true);
    _hdr = stbi_is_hdr(filename.c_str());
    if(_hdr)
        loadHDR(filename);
    else
        loadLDR(filename);
    _filename = filename;
}

void Image::store(const std::string& filename)
{
    stbi_flip_vertically_on_write(true);
    std::string file_ending = filename.substr(filename.find_last_of(".") + 1);
    if(file_ending == "png")
    {
        stbi_write_png(filename.c_str(),
                       (int)_width,
                       (int)_height,
                       (int)_channels,
                       (const void*)_img_data.data_ptr(),
                       _channels * _width * sizeof(uint8_t));
    }
    else if(file_ending == "bmp")
    {
        stbi_write_bmp(filename.c_str(), (int)_width, (int)_height, (int)_channels, (const void*)_img_data.data_ptr());
    }
    else if(file_ending == "tga")
    {
        stbi_write_tga(filename.c_str(), (int)_width, (int)_height, (int)_channels, (const void*)_img_data.data_ptr());
    }
    else if(file_ending == "jpg")
    {
        stbi_write_jpg(filename.c_str(),
                       (int)_width,
                       (int)_height,
                       (int)_channels,
                       (const void*)_img_data.data_ptr(),
                       100);
    }
    else if(file_ending == "hdr")
    {
        stbi_write_hdr(filename.c_str(), (int)_width, (int)_height, (int)_channels, (const float*)_img_data.data_ptr());
    }
}

void Image::applyGamma(const float gamma)
{
    if(_hdr)
    {
        _img_data = _img_data.pow(gamma);
    }
    else
    {
        _img_data = ((_img_data.to(torch::kFloat32) / 255.0f).pow(gamma) * 255.0f).to(torch::kUInt8);
    }
}

void Image::setData(uint8_t* data, uint32_t width, uint32_t height, uint32_t channels)
{
    _width    = width;
    _height   = height;
    _channels = channels;

    _img_data = atcg::createHostTensorFromPointer<uint8_t>(data, {(int)height, (int)width, (int)channels}).clone();
}

void Image::setData(float* data, uint32_t width, uint32_t height, uint32_t channels)
{
    _width    = width;
    _height   = height;
    _hdr      = true;
    _channels = channels;

    _img_data = atcg::createHostTensorFromPointer<float>(data, {(int)height, (int)width, (int)channels}).clone();
}

void Image::setData(const torch::Tensor& tensor)
{
    TORCH_CHECK_EQ(tensor.scalar_type() == torch::kFloat32 || tensor.scalar_type() == torch::kUInt8, true);

    _img_data = tensor;
    _width    = tensor.size(1);
    _height   = tensor.size(0);
    _channels = tensor.size(2);
    _hdr      = tensor.dtype() == torch::kFloat32;
}

void Image::loadLDR(const std::string& filename)
{
    uint8_t* data = stbi_load(filename.c_str(), (int*)&_width, (int*)&_height, (int*)&_channels, 0);

    if(data == nullptr)
    {
        ATCG_ERROR("Image::loadLDR: Error loading image {0}", filename);
    }

    setData(data, _width, _height, _channels);
    stbi_image_free(data);
}

void Image::loadHDR(const std::string& filename)
{
    _hdr        = true;
    float* data = stbi_loadf(filename.c_str(), (int*)&_width, (int*)&_height, (int*)&_channels, 0);

    if(data == nullptr)
    {
        ATCG_ERROR("Image::loadLDR: Error loading image {0}", filename);
    }

    setData(data, _width, _height, _channels);
    stbi_image_free(data);
}


namespace IO
{
atcg::ref_ptr<Image> imread(const std::string& filename, const float gamma)
{
    atcg::ref_ptr<Image> img = atcg::make_ref<Image>();

    img->load(filename);
    if(gamma != 1.0f) img->applyGamma(gamma);

    return img;
}
void imwrite(const atcg::ref_ptr<Image>& image, const std::string& filename, const float gamma)
{
    if(gamma != 1.0f) image->applyGamma(gamma);
    image->store(filename);
}
}    // namespace IO
}    // namespace atcg