#include <DataStructure/JPEGDecoder.h>

#include <DataStructure/TorchUtils.h>

#include <cstdlib>

namespace atcg
{

class JPEGDecoder::Impl
{
public:
    Impl(uint32_t num_images, uint32_t img_width, uint32_t img_height);

    ~Impl();

    void allocateBuffers();

    // Rendering buffers
    torch::Tensor output_tensor;
    torch::Tensor intermediate_tensor;

    uint32_t num_images;
    uint32_t img_width;
    uint32_t img_height;
};

JPEGDecoder::Impl::Impl(uint32_t num_images, uint32_t img_width, uint32_t img_height)
{
    this->num_images = num_images;
    this->img_width  = img_width;
    this->img_height = img_height;
    allocateBuffers();
}

JPEGDecoder::Impl::~Impl() {}

void JPEGDecoder::Impl::allocateBuffers()
{
    output_tensor = torch::zeros({num_images, img_height, img_width, 3}, atcg::TensorOptions::uint8HostOptions());
}

JPEGDecoder::JPEGDecoder(uint32_t num_images, uint32_t img_width, uint32_t img_height)
{
    impl = std::make_unique<Impl>(num_images, img_width, img_height);
}

JPEGDecoder::~JPEGDecoder() {}

torch::Tensor JPEGDecoder::decompressImages(const std::vector<std::string>& filenames)
{
    return decompressImages(filenames, torch::ones(impl->num_images, atcg::TensorOptions::int32HostOptions()));
}

torch::Tensor JPEGDecoder::decompressImages(const std::vector<std::string>& filenames, const torch::Tensor& host_valid)
{
    // TODO
    return impl->output_tensor;
}

void JPEGDecoder::copyToOutput(const atcg::ref_ptr<Texture3D>& texture)
{
    texture->setData(impl->output_tensor);
}

void JPEGDecoder::copyToOutput(atcg::textureArray output_texture)
{
    ATCG_ERROR("JPEGDecoder::copyToOutput(atcg::textureArray) not implemented on CPU backend");
}
}    // namespace atcg