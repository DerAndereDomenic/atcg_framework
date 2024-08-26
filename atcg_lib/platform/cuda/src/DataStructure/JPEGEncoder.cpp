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

class JPEGEncoder::Impl
{
public:
    Impl();

    ~Impl();

    void allocateBuffers();
    void initializeNVJPEG();
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

    cudaStream_t encoding_stream;
};

JPEGEncoder::Impl::Impl()
{
    allocateBuffers();
    initializeNVJPEG();
}

JPEGEncoder::Impl::~Impl()
{
    deinitializeNVJPEG();
    CUDA_SAFE_CALL(cudaStreamDestroy(encoding_stream));
}

void JPEGEncoder::Impl::allocateBuffers()
{
    CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&encoding_stream, cudaStreamNonBlocking));
}

void JPEGEncoder::Impl::initializeNVJPEG()
{
    NVJPEG_SAFE_CALL(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &pinned_allocator, flags, &nvjpeg_handle));
    NVJPEG_SAFE_CALL(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state));
    NVJPEG_SAFE_CALL(nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, encoding_stream));
    NVJPEG_SAFE_CALL(nvjpegEncoderParamsCreate(nvjpeg_handle, &encode_params, encoding_stream));

    NVJPEG_SAFE_CALL(nvjpegEncoderParamsSetQuality(encode_params, 90, encoding_stream));
    NVJPEG_SAFE_CALL(nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_444, encoding_stream));
    // NVJPEG_SAFE_CALL(nvjpegEncoderParamsSetOptimizedHuffman(encode_params, 0, NULL));
}

torch::Tensor JPEGEncoder::Impl::compressImage(const torch::Tensor& img)
{
    uint32_t num_channels          = img.size(-1);
    torch::Tensor intermediate_img = img;
    if(num_channels == 4)
    {
        intermediate_img = img.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3)})
                               .clone()
                               .contiguous();
    }
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
                                       encoding_stream));
    CUDA_SAFE_CALL(cudaStreamSynchronize(encoding_stream));

    size_t length;
    NVJPEG_SAFE_CALL(nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, NULL, &length, encoding_stream));
    CUDA_SAFE_CALL(cudaStreamSynchronize(encoding_stream));
    ATCG_TRACE(length);

    torch::Tensor output = torch::empty({(int)length}, atcg::TensorOptions::uint8HostOptions());
    NVJPEG_SAFE_CALL(nvjpegEncodeRetrieveBitstream(nvjpeg_handle,
                                                   encoder_state,
                                                   output.data_ptr<unsigned char>(),
                                                   &length,
                                                   encoding_stream));
    CUDA_SAFE_CALL(cudaStreamSynchronize(encoding_stream));
    return output;
}

void JPEGEncoder::Impl::deinitializeNVJPEG()
{
    NVJPEG_SAFE_CALL(nvjpegEncoderParamsDestroy(encode_params));
    NVJPEG_SAFE_CALL(nvjpegEncoderStateDestroy(encoder_state));
    NVJPEG_SAFE_CALL(nvjpegJpegStateDestroy(nvjpeg_state));
    NVJPEG_SAFE_CALL(nvjpegDestroy(nvjpeg_handle));
}

JPEGEncoder::JPEGEncoder()
{
    impl = std::make_unique<Impl>();
}

JPEGEncoder::~JPEGEncoder() {}

torch::Tensor JPEGEncoder::compress(const torch::Tensor& img)
{
    return impl->compressImage(img);
}
}    // namespace atcg