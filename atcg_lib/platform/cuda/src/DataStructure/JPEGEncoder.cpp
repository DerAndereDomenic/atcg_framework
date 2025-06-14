#include <DataStructure/JPEGEncoder.h>

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
        std::cout << "NVJPEG error at " << file << ":" << line << " code=(" << static_cast<unsigned int>(error)
                  << ") \"" << func << "\" \n";
    }
}

inline static int dev_malloc(void** p, size_t s)
{
    try
    {
        auto stream = torch::cuda::getCurrentCUDAStream();
        *p          = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(s, stream);
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

class JPEGEncoder::Impl
{
public:
    Impl(bool flip_vertically, JPEGBackend backend);

    ~Impl();

    void allocateBuffers();
    void initializeNVJPEG(JPEGBackend backend);
    torch::Tensor compressImage(const torch::Tensor& tensor);

    void deinitializeNVJPEG();

    // NVJPEG States
    nvjpegDevAllocator_t dev_allocator       = {&detail::dev_malloc, &detail::dev_free};
    nvjpegPinnedAllocator_t pinned_allocator = {&detail::host_malloc, &detail::host_free};
    int flags                                = 0;
    nvjpegHandle_t nvjpeg_handle;
    nvjpegJpegState_t nvjpeg_state;
    nvjpegEncoderState_t encoder_state;
    nvjpegEncoderParams_t encode_params;
    nvjpegInputFormat_t fmt = NVJPEG_INPUT_RGBI;

    bool flip_vertically = true;
};

JPEGEncoder::Impl::Impl(bool flip_vertically, JPEGBackend backend)
{
    this->flip_vertically = flip_vertically;

    // This initializes the cache allocator
    torch::empty({1}, atcg::TensorOptions::floatDeviceOptions());
    torch::empty({1}, atcg::TensorOptions::floatHostOptions());

    allocateBuffers();
    initializeNVJPEG(backend);
}

JPEGEncoder::Impl::~Impl()
{
    deinitializeNVJPEG();
}

void JPEGEncoder::Impl::allocateBuffers() {}

void JPEGEncoder::Impl::initializeNVJPEG(JPEGBackend backend)
{
    cudaStream_t torch_stream = c10::cuda::getCurrentCUDAStream();

    nvjpegBackend_t nvjpeg_backend =
        backend == JPEGBackend::SOFTWARE ? NVJPEG_BACKEND_DEFAULT : NVJPEG_BACKEND_HARDWARE;

    NVJPEG_SAFE_CALL(nvjpegCreateEx(nvjpeg_backend, &dev_allocator, &pinned_allocator, flags, &nvjpeg_handle));
    NVJPEG_SAFE_CALL(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state));
    NVJPEG_SAFE_CALL(nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, torch_stream));
    NVJPEG_SAFE_CALL(nvjpegEncoderParamsCreate(nvjpeg_handle, &encode_params, torch_stream));

    NVJPEG_SAFE_CALL(nvjpegEncoderParamsSetQuality(encode_params, 90, torch_stream));
    NVJPEG_SAFE_CALL(nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_444, torch_stream));
    // NVJPEG_SAFE_CALL(nvjpegEncoderParamsSetOptimizedHuffman(encode_params, 0, NULL));
}

torch::Tensor JPEGEncoder::Impl::compressImage(const torch::Tensor& img)
{
    cudaStream_t torch_stream      = c10::cuda::getCurrentCUDAStream();
    uint32_t num_channels          = img.size(-1);
    torch::Tensor intermediate_img = img;

    if(num_channels == 4)
    {
        intermediate_img =
            img.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3)}).clone();
    }

    if(flip_vertically)
    {
        intermediate_img = intermediate_img.flip(0);
    }

    intermediate_img = intermediate_img.contiguous();

    // Stream sync to ensure PyTorch has finished preparing data

    nvjpegImage_t source = {};
    source.pitch[0]      = 3 * intermediate_img.size(1);
    source.channel[0]    = intermediate_img.data_ptr<unsigned char>();
    NVJPEG_SAFE_CALL(nvjpegEncodeImage(nvjpeg_handle,
                                       encoder_state,
                                       encode_params,
                                       &source,
                                       fmt,
                                       intermediate_img.size(1),
                                       intermediate_img.size(0),
                                       torch_stream));

    size_t length;
    NVJPEG_SAFE_CALL(nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, NULL, &length, torch_stream));

    torch::Tensor output = torch::empty({(int)length}, atcg::TensorOptions::uint8HostOptions());
    NVJPEG_SAFE_CALL(nvjpegEncodeRetrieveBitstream(nvjpeg_handle,
                                                   encoder_state,
                                                   output.data_ptr<unsigned char>(),
                                                   &length,
                                                   torch_stream));

    return output;
}

void JPEGEncoder::Impl::deinitializeNVJPEG()
{
    NVJPEG_SAFE_CALL(nvjpegEncoderParamsDestroy(encode_params));
    NVJPEG_SAFE_CALL(nvjpegEncoderStateDestroy(encoder_state));
    NVJPEG_SAFE_CALL(nvjpegJpegStateDestroy(nvjpeg_state));
    NVJPEG_SAFE_CALL(nvjpegDestroy(nvjpeg_handle));
}

JPEGEncoder::JPEGEncoder(bool flip_vertically, JPEGBackend backend)
{
    impl = std::make_unique<Impl>(flip_vertically, backend);
}

JPEGEncoder::~JPEGEncoder() {}

torch::Tensor JPEGEncoder::compress(const torch::Tensor& img)
{
    return impl->compressImage(img);
}
}    // namespace atcg