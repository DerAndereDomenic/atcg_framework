#include <DataStructure/JPEGDecoder.h>

#include <DataStructure/TorchUtils.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <cstdlib>
#include <nvjpeg.h>

namespace atcg
{
namespace detail
{

inline void check_nvjpeg(nvjpegStatus_t error, char const* const func, const char* const file, int const line)
{
    if(error != NVJPEG_STATUS_SUCCESS)
    {
        ATCG_ERROR("NVJPEG error at {0}:{1} code=({3}) \"{4}\" \n", file, line, static_cast<unsigned int>(error), func);
    }
}

int dev_malloc(void** p, size_t s)
{
    try
    {
        *p = c10::cuda::CUDACachingAllocator::raw_alloc(s);
        return EXIT_SUCCESS;
    }
    catch(const std::exception& e)
    {
        return EXIT_FAILURE;
    }
}

int dev_free(void* p)
{
    try
    {
        c10::cuda::CUDACachingAllocator::raw_delete(p);
        return EXIT_SUCCESS;
    }
    catch(const std::exception& e)
    {
        return EXIT_FAILURE;
    }
}

int host_malloc(void** p, size_t s, unsigned int f)
{
    return (int)cudaHostAlloc(p, s, f);
}

int host_free(void* p)
{
    return (int)cudaFreeHost(p);
}
}    // namespace detail

#ifdef NDEBUG
    #define NVJPEG_SAFE_CALL(val) val
#else
    #define NVJPEG_SAFE_CALL(val) detail::check_nvjpeg((val), #val, __FILE__, __LINE__)
#endif

class JPEGDecoder::Impl
{
public:
    Impl(uint32_t num_images, uint32_t img_width, uint32_t img_height);

    ~Impl();

    void allocateBuffers();
    void initializeNVJPEG();
    void loadFiles(const std::vector<std::string>& filenames, const torch::Tensor& valid);
    void decompressImages();
    void copyImagesToOutput(atcg::textureArray texture);

    void deinitializeNVJPEG();

    // Rendering buffers
    torch::Tensor output_tensor;
    torch::Tensor intermediate_tensor;

    // File buffers
    std::vector<std::vector<char>> file_data;
    std::vector<size_t> file_lengths;
    std::vector<const unsigned char*> raw_inputs;
    std::vector<nvjpegImage_t> output_images;

    // NVJPEG States
    nvjpegDevAllocator_t dev_allocator       = {&detail::dev_malloc, &detail::dev_free};
    nvjpegPinnedAllocator_t pinned_allocator = {&detail::host_malloc, &detail::host_free};
    int flags                                = 0;
    nvjpegHandle_t nvjpeg_handle;
    nvjpegJpegState_t nvjpeg_state;
    nvjpegOutputFormat_t fmt = NVJPEG_OUTPUT_RGBI;

    uint32_t num_images;
    uint32_t img_width;
    uint32_t img_height;

    cudaStream_t decoding_stream;
};

JPEGDecoder::Impl::Impl(uint32_t num_images, uint32_t img_width, uint32_t img_height)
{
    this->num_images = num_images;
    this->img_width  = img_width;
    this->img_height = img_height;
    allocateBuffers();
    initializeNVJPEG();
}

JPEGDecoder::Impl::~Impl()
{
    deinitializeNVJPEG();
    CUDA_SAFE_CALL(cudaStreamDestroy(decoding_stream));
}

void JPEGDecoder::Impl::allocateBuffers()
{
    file_data.resize(num_images);
    // for(int i = 0; i < num_images; ++i) { file_data[i] = std::vector<char>(); }
    file_lengths.resize(num_images);
    raw_inputs.resize(num_images);

    output_tensor = torch::zeros({num_images, img_height, img_width, 3}, atcg::TensorOptions::uint8DeviceOptions());

    CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&decoding_stream, cudaStreamNonBlocking));
}

void JPEGDecoder::Impl::initializeNVJPEG()
{
    NVJPEG_SAFE_CALL(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &pinned_allocator, flags, &nvjpeg_handle));
    NVJPEG_SAFE_CALL(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state));
    NVJPEG_SAFE_CALL(nvjpegDecodeBatchedInitialize(nvjpeg_handle, nvjpeg_state, num_images, 1, fmt));

    // Prepare buffers
    nvjpegChromaSubsampling_t subsampling;

    output_images.resize(num_images);
    std::memset(output_images.data(), 0, sizeof(nvjpegImage_t) * num_images);

    for(int i = 0; i < num_images; i++)
    {
        output_images[i].pitch[0]   = 3 * img_width;
        output_images[i].channel[0] = (unsigned char*)output_tensor.index({i, 0, 0, 0}).data_ptr();
    }
}

void JPEGDecoder::Impl::loadFiles(const std::vector<std::string>& filenames, const torch::Tensor& valid)
{
    torch::Tensor host_valid = valid.to(torch::kCPU);
    int index                = 0;
    for(uint32_t i = 0; i < filenames.size(); ++i)
    {
        if(host_valid.index({(int)i}).item<int>() == 0)
        {
            continue;
        }

        std::ifstream input(filenames[i], std::ios::in | std::ios::binary | std::ios::ate);

        // Get the size
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        // resize if buffer is too small
        if(file_data[index].size() < file_size)
        {
            file_data[index].resize(file_size);
        }
        if(!input.read(file_data[index].data(), file_size))
        {
            ATCG_ERROR("JPEGDecoder: Cannot read from file: {0}", filenames[i]);
        }
        file_lengths[index] = file_size;

        raw_inputs[index] = (const unsigned char*)file_data[index].data();

        ++index;
        if(index >= num_images) break;
    }
}

void JPEGDecoder::Impl::decompressImages()
{
    NVJPEG_SAFE_CALL(nvjpegDecodeBatched(nvjpeg_handle,
                                         nvjpeg_state,
                                         raw_inputs.data(),
                                         file_lengths.data(),
                                         output_images.data(),
                                         decoding_stream));
    CUDA_SAFE_CALL(cudaStreamSynchronize(decoding_stream));
}

void JPEGDecoder::Impl::copyImagesToOutput(atcg::textureArray texture)
{
    intermediate_tensor =
        torch::cat(
            {output_tensor,
             torch::full({num_images, img_height, img_width, 1}, 255, atcg::TensorOptions::uint8DeviceOptions())},
            -1)
            .contiguous();

    cudaChannelFormatDesc desc = {};
    cudaExtent ext             = {};
    unsigned int array_flags   = 0;

    CUDA_SAFE_CALL(cudaArrayGetInfo(&desc, &ext, &array_flags, texture));

    cudaMemcpy3DParms p = {0};
    p.dstArray          = texture;
    p.kind              = cudaMemcpyDeviceToDevice;
    p.srcPtr.ptr        = intermediate_tensor.contiguous().data_ptr();
    p.srcPtr.pitch      = ext.width * 4;
    p.srcPtr.xsize      = ext.width;
    p.srcPtr.ysize      = ext.height;
    p.extent            = ext;

    CUDA_SAFE_CALL(cudaMemcpy3D(&p));
}

void JPEGDecoder::Impl::deinitializeNVJPEG()
{
    NVJPEG_SAFE_CALL(nvjpegJpegStateDestroy(nvjpeg_state));
    NVJPEG_SAFE_CALL(nvjpegDestroy(nvjpeg_handle));
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

torch::Tensor JPEGDecoder::decompressImages(const std::vector<std::string>& filenames, const torch::Tensor& valid)
{
    impl->loadFiles(filenames, valid);
    impl->decompressImages();

    return impl->output_tensor;
}

void JPEGDecoder::copyToOutput(const atcg::ref_ptr<Texture3D>& texture)
{
    TORCH_CHECK_EQ(texture->width(), impl->img_width);
    TORCH_CHECK_EQ(texture->height(), impl->img_height);
    TORCH_CHECK_EQ(texture->depth(), impl->num_images);

    auto output_texture = texture->getTextureArray();
    copyToOutput(output_texture);
}

void JPEGDecoder::copyToOutput(atcg::textureArray output_texture)
{
    impl->copyImagesToOutput(output_texture);
}
}