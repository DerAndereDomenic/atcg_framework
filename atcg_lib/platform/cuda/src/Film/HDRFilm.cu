#include <Film/HDRFilm.h>
#include <DataStructure/TorchUtils.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <c10/cuda/CUDAGuard.h>

namespace atcg
{

namespace detail
{
ATCG_GLOBAL void
develop_kernel(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> accumulation_buffer,
               const uint32_t width,
               const uint32_t height,
               torch::PackedTensorAccessor32<uint8_t, 3, torch::RestrictPtrTraits> output_image)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for(auto tid = id; tid < width * height; tid += num_threads)
    {
        uint32_t x = static_cast<uint32_t>(tid % width);
        uint32_t y = static_cast<uint32_t>(tid / width);

        glm::vec3 radiance =
            glm::vec3(accumulation_buffer[y][x][0], accumulation_buffer[y][x][1], accumulation_buffer[y][x][2]);

        radiance = glm::vec3(1.0f) - glm::exp(-radiance);

        glm::vec3 sRGB = atcg::Color::lRGB_to_sRGB(radiance);

        sRGB.x = glm::min(glm::max(sRGB.x, 0.0f), 1.0f);
        sRGB.y = glm::min(glm::max(sRGB.y, 0.0f), 1.0f);
        sRGB.z = glm::min(glm::max(sRGB.z, 0.0f), 1.0f);

        glm::u8vec3 quantized = atcg::Color::quantize(sRGB);

        output_image[y][x][0] = quantized.x;
        output_image[y][x][1] = quantized.y;
        output_image[y][x][2] = quantized.z;
        output_image[y][x][3] = 255;
    }
}
}    // namespace detail

HDRFilm::HDRFilm() : Film()
{
    _width  = 0;
    _height = 0;

    HDRFilmData hdr_film_data;
    hdr_film_data.width               = 0;
    hdr_film_data.height              = 0;
    hdr_film_data.accumulation_buffer = nullptr;

    _hdr_film_data.upload(&hdr_film_data);
}

HDRFilm::HDRFilm(const atcg::Dictionary& dict) : Film(dict)
{
    uint32_t width  = dict.getValue<uint32_t>("width");
    uint32_t height = dict.getValue<uint32_t>("height");

    resize(width, height);
}

HDRFilm::~HDRFilm() {}

void HDRFilm::onImGuiRender() {}

torch::Tensor HDRFilm::develop() const
{
    torch::Tensor output_image =
        torch::zeros({(int64_t)_height, (int64_t)_width, 4}, atcg::TensorOptions::uint8DeviceOptions());

    auto device = output_image.device();

    {
        at::cuda::CUDAGuard device_guard(device);
        const auto stream = at::cuda::getCurrentCUDAStream();

        const int threads_per_block = 128;
        dim3 grid;
        at::cuda::getApplyGrid(_height * _width, grid, device.index(), threads_per_block);
        dim3 threads = at::cuda::getApplyBlock(threads_per_block);

        detail::develop_kernel<<<grid, threads, 0, stream>>>(
            _accumulation_buffer.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            _width,
            _height,
            output_image.packed_accessor32<uint8_t, 3, torch::RestrictPtrTraits>());

        AT_CUDA_CHECK(cudaGetLastError());
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    return output_image;
}

void HDRFilm::resize(uint32_t width, uint32_t height)
{
    _width  = width;
    _height = height;
    _accumulation_buffer =
        torch::zeros({(int64_t)height, (int64_t)width, 3}, atcg::TensorOptions::floatDeviceOptions());

    HDRFilmData hdr_film_data;
    hdr_film_data.width               = width;
    hdr_film_data.height              = height;
    hdr_film_data.accumulation_buffer = (glm::vec3*)_accumulation_buffer.data_ptr();

    _hdr_film_data.upload(&hdr_film_data);
}

void HDRFilm::clear()
{
    _accumulation_buffer.zero_();
}

void HDRFilm::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                 const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_film_filename = "./bin/HDRFilm_ptx.ptx";
    auto add_sample_prog_group =
        pipeline->addCallableShader({ptx_film_filename, "__direct_callable__add_sample_hdrfilm"});
    auto get_width_prog_group =
        pipeline->addCallableShader({ptx_film_filename, "__direct_callable__get_width_hdrfilm"});
    auto get_height_prog_group =
        pipeline->addCallableShader({ptx_film_filename, "__direct_callable__get_height_hdrfilm"});
    uint32_t add_sampled_idx = sbt->addCallableEntry(add_sample_prog_group, _hdr_film_data.get());
    uint32_t get_width_idx   = sbt->addCallableEntry(get_width_prog_group, _hdr_film_data.get());
    uint32_t get_height_idx  = sbt->addCallableEntry(get_height_prog_group, _hdr_film_data.get());

    FilmVPtrTable vptr_table;
    vptr_table.addSampleCallIndex = add_sampled_idx;
    vptr_table.getWidthCallIndex  = get_width_idx;
    vptr_table.getHeightCallIndex = get_height_idx;

    _vptr_table.upload(&vptr_table);
    markInitialized();
}
}    // namespace atcg