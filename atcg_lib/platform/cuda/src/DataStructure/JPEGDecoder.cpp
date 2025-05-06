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
        std::cout << "NVJPEG error at " << file << ":" << line << "code=(" << static_cast<unsigned int>(error) << ") \""
                  << func << "\" \n";
    }
}

inline static int dev_malloc(void** p, size_t s)
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

inline static int dev_free(void* p)
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

inline static int host_malloc(void** p, size_t s, unsigned int f)
{
    return (int)cudaHostAlloc(p, s, f);
}

inline static int host_free(void* p)
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
    Impl(uint32_t num_images, uint32_t img_width, uint32_t img_height, bool flip_vertically, JPEGBackend backend);

    ~Impl();

    void allocateBuffers();
    void initializeNVJPEG(JPEGBackend backend);
    void loadFiles(const std::vector<std::vector<uint8_t>>& jpeg_files);
    void decompressImages();
    void copyImagesToOutput(atcg::textureArray texture);

    void deinitializeNVJPEG();

    // Rendering buffers
    torch::Tensor output_tensor;
    torch::Tensor data_tensor;
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

    bool flip_vertically = true;
};

JPEGDecoder::Impl::Impl(uint32_t num_images,
                        uint32_t img_width,
                        uint32_t img_height,
                        bool flip_vertically,
                        JPEGBackend backend)
{
    this->num_images      = num_images;
    this->img_width       = img_width;
    this->img_height      = img_height;
    this->flip_vertically = flip_vertically;

    // This initializes the cache allocator
    torch::empty({1}, atcg::TensorOptions::floatDeviceOptions());
    torch::empty({1}, atcg::TensorOptions::floatHostOptions());

    allocateBuffers();
    initializeNVJPEG(backend);
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

    data_tensor = torch::zeros({num_images, img_height, img_width, 3}, atcg::TensorOptions::uint8DeviceOptions());

    CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&decoding_stream, cudaStreamNonBlocking));
}

void JPEGDecoder::Impl::initializeNVJPEG(JPEGBackend backend)
{
    nvjpegBackend_t nvjpeg_backend =
        backend == JPEGBackend::SOFTWARE ? NVJPEG_BACKEND_DEFAULT : NVJPEG_BACKEND_HARDWARE;

    NVJPEG_SAFE_CALL(nvjpegCreateEx(nvjpeg_backend, &dev_allocator, &pinned_allocator, flags, &nvjpeg_handle));
    NVJPEG_SAFE_CALL(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state));
    NVJPEG_SAFE_CALL(nvjpegDecodeBatchedInitialize(nvjpeg_handle, nvjpeg_state, num_images, 1, fmt));

    // Prepare buffers
    nvjpegChromaSubsampling_t subsampling;

    output_images.resize(num_images);
    std::memset(output_images.data(), 0, sizeof(nvjpegImage_t) * num_images);

    for(int i = 0; i < num_images; i++)
    {
        output_images[i].pitch[0]   = 3 * img_width;
        output_images[i].channel[0] = (unsigned char*)data_tensor.index({i, 0, 0, 0}).data_ptr();
    }
}

void JPEGDecoder::Impl::loadFiles(const std::vector<std::vector<uint8_t>>& jpeg_files)
{
    for(uint32_t i = 0; i < jpeg_files.size(); ++i)
    {
        file_lengths[i] = jpeg_files[i].size();

        raw_inputs[i] = (const unsigned char*)jpeg_files[i].data();
    }
}

void JPEGDecoder::Impl::decompressImages()
{
    cudaStream_t torch_stream = c10::cuda::getCurrentCUDAStream();
    cudaEvent_t torch_ready_event;
    CUDA_SAFE_CALL(cudaEventCreateWithFlags(&torch_ready_event, cudaEventDisableTiming));
    CUDA_SAFE_CALL(cudaEventRecord(torch_ready_event, torch_stream));
    CUDA_SAFE_CALL(cudaStreamWaitEvent(decoding_stream, torch_ready_event, 0));

    NVJPEG_SAFE_CALL(nvjpegDecodeBatched(nvjpeg_handle,
                                         nvjpeg_state,
                                         raw_inputs.data(),
                                         file_lengths.data(),
                                         output_images.data(),
                                         decoding_stream));
    CUDA_SAFE_CALL(cudaStreamSynchronize(decoding_stream));

    // Unfortunetly, nvjpeg does not support inplace flipping of decoded images.
    // We therefore create a copy of the data although it is not super efficient...
    if(flip_vertically)
    {
        output_tensor = data_tensor.flip(1);
    }
    else
    {
        output_tensor = data_tensor.clone();
    }

    CUDA_SAFE_CALL(cudaEventDestroy(torch_ready_event));
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
    p.srcPtr.ptr        = intermediate_tensor.data_ptr();
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

JPEGDecoder::JPEGDecoder(uint32_t num_images,
                         uint32_t img_width,
                         uint32_t img_height,
                         bool flip_vertically,
                         JPEGBackend backend)
{
    impl = std::make_unique<Impl>(num_images, img_width, img_height, flip_vertically, backend);
}

JPEGDecoder::~JPEGDecoder() {}

torch::Tensor JPEGDecoder::decompressImages(const std::vector<std::vector<uint8_t>>& jpeg_files)
{
    impl->loadFiles(jpeg_files);
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
}    // namespace atcg