#include <DataStructure/Image.h>

#include <stb_image.h>
#include <stb_image_write.h>

namespace atcg
{

Image::Image(const uint8_t* data, uint32_t width, uint32_t height, uint32_t channels)
{
    _width  = width;
    _height = height;

    if(channels == 1 || channels == 4)
    {
        _img_data = (uint8_t*)malloc(width * height * channels *
                                     sizeof(uint8_t));    // We use malloc here for compatibility with stbi
        memcpy(_img_data, data, width * height * channels * sizeof(uint8_t));
        _channels = channels;
    }
    else
    {
        // Padding
        _img_data = (uint8_t*)malloc(width * height * 4 * sizeof(uint8_t));

        for(uint32_t i = 0; i < width * height; ++i)    // Iterate through each rg or rgb pixel
        {
            uint8_t padded[4] = {data[channels * i],
                                 data[channels * i + 1],
                                 channels == 3 ? data[channels * i + 2] : 0,
                                 255};
            memcpy(_img_data + channels * i, padded, sizeof(padded));
        }

        _channels = 4;
    }
}

Image::Image(const float* data, uint32_t width, uint32_t height, uint32_t channels)
{
    _width  = width;
    _height = height;
    _hdr    = true;

    if(channels == 1 || channels == 4)
    {
        _img_data = (uint8_t*)malloc(width * height * channels *
                                     sizeof(float));    // We use malloc here for compatibility with stbi
        memcpy(_img_data, data, width * height * channels * sizeof(float));
        _channels = channels;
    }
    else
    {
        _img_data = (uint8_t*)malloc(width * height * 4 * sizeof(float));

        float* data_float = (float*)data;

        for(uint32_t i = 0; i < width * height; ++i)    // Iterate through each rg or rgb pixel
        {
            float padded[4] = {data_float[channels * i],
                               data_float[channels * i + 1],
                               channels == 3 ? data_float[channels * i + 2] : 0.0f,
                               1.0f};
            memcpy(data_float + channels * i, padded, sizeof(padded));
        }
        // Padding
        _channels = 4;
    }
}

Image::~Image()
{
    if(_img_data)
    {
        free(_img_data);
        uint8_t* _img_data;
        uint32_t _width    = 0;
        uint32_t _height   = 0;
        uint32_t _channels = 0;
        _hdr               = false;
    }
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
                       (const void*)_img_data,
                       _channels * _width * sizeof(uint8_t));
    }
    else if(file_ending == "bmp")
    {
        stbi_write_bmp(filename.c_str(), (int)_width, (int)_height, (int)_channels, (const void*)_img_data);
    }
    else if(file_ending == "tga")
    {
        stbi_write_tga(filename.c_str(), (int)_width, (int)_height, (int)_channels, (const void*)_img_data);
    }
    else if(file_ending == "jpg")
    {
        stbi_write_jpg(filename.c_str(), (int)_width, (int)_height, (int)_channels, (const void*)_img_data, 100);
    }
    else if(file_ending == "hdr")
    {
        stbi_write_hdr(filename.c_str(), (int)_width, (int)_height, (int)_channels, (const float*)_img_data);
    }
}

void Image::applyGamma(const float gamma)
{
    if(stbi_is_hdr(_filename.c_str()))
    {
        float* data = (float*)_img_data;
        for(uint32_t i = 0; i < _width * _height * _channels; ++i) { data[i] = glm::pow(data[i], gamma); }
    }
    else
    {
        uint8_t* data = _img_data;
        for(uint32_t i = 0; i < _width * _height * _channels; ++i)
        {
            data[i] = (uint8_t)(255.0f * glm::pow((float)data[i] / 255.0f, gamma));
        }
    }
}

void Image::setData(const uint8_t* data)
{
    size_t size = stbi_is_hdr(_filename.c_str()) ? sizeof(float) : sizeof(uint8_t);
    memcpy((void*)_img_data, (const void*)data, _width * _height * _channels * size);
}

void Image::loadLDR(const std::string& filename)
{
    stbi_info(filename.c_str(), (int*)&_width, (int*)&_height, (int*)&_channels);

    if(_channels == 1) { _img_data = stbi_load(filename.c_str(), (int*)&_width, (int*)&_height, (int*)&_channels, 0); }
    else
    {
        _img_data = stbi_load(filename.c_str(), (int*)&_width, (int*)&_height, (int*)&_channels, 4);
        _channels = 4;
    }
}

void Image::loadHDR(const std::string& filename)
{
    stbi_info(filename.c_str(), (int*)&_width, (int*)&_height, (int*)&_channels);

    if(_channels == 1)
    {
        _img_data = (uint8_t*)stbi_loadf(filename.c_str(), (int*)&_width, (int*)&_height, (int*)&_channels, 0);
    }
    else
    {
        _img_data = (uint8_t*)stbi_loadf(filename.c_str(), (int*)&_width, (int*)&_height, (int*)&_channels, 4);
        _channels = 4;
    }
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