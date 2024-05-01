#include <Renderer/Pathtracer.h>

#include <DataStructure/Timer.h>

#include <Scene/Entity.h>
#include <Scene/Components.h>

#include <Renderer/Common.h>

#include <thread>
#include <mutex>
#include <execution>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <Renderer/RaytracingPipeline.h>
#include <Renderer/ShaderBindingTable.h>


#include <Renderer/PathtracingShader.h>

namespace atcg
{

Pathtracer* Pathtracer::s_pathtracer = new Pathtracer;

class Pathtracer::Impl
{
public:
    Impl() {}
    ~Impl() {}

    void start();
    void stop();
    void worker();
    void resize(const uint32_t width, const uint32_t height);

    // OptiX
    OptixDeviceContext context = nullptr;
    atcg::ref_ptr<RayTracingPipeline> raytracing_pipeline;
    atcg::ref_ptr<ShaderBindingTable> sbt;

    glm::u8vec4* output_buffer;
    uint8_t swap_index = 1;
    torch::Tensor swap_chain_buffer;
    bool dirty = false;
    std::mutex swap_chain_mutex;

    atcg::ref_ptr<atcg::Texture2D> output_texture;

    std::thread worker_thread;

    std::atomic_bool running = false;

    atcg::ref_ptr<OptixRaytracingShader> shader = nullptr;
    uint32_t num_samples;
};

void Pathtracer::Impl::worker()
{
    for(int subframe = 0; subframe < num_samples; ++subframe)
    {
        Timer frame_time;

        shader->generateRays(
            raytracing_pipeline,
            sbt,
            atcg::createDeviceTensorFromPointer<uint8_t>((uint8_t*)output_buffer,
                                                         {output_texture->height(), output_texture->width(), 4}));

        ATCG_TRACE(frame_time.elapsedMillis());

        // Perform swap
        {
            std::lock_guard guard(swap_chain_mutex);
            output_buffer = (glm::u8vec4*)swap_chain_buffer.index({swap_index}).data_ptr();
            swap_index    = (swap_index + 1) % 2;
            dirty         = true;
        }

        if(!running) break;
    }

    running = false;
}

Pathtracer::Pathtracer() {}

Pathtracer::~Pathtracer() {}

void Pathtracer::init()
{
    s_pathtracer->impl = std::make_unique<Impl>();

    // Just create some dummy data
    s_pathtracer->impl->swap_chain_buffer = torch::zeros({2, 1, 1, 4}, atcg::TensorOptions::uint8DeviceOptions());
    s_pathtracer->impl->output_buffer     = (glm::u8vec4*)s_pathtracer->impl->swap_chain_buffer.index({0}).data_ptr();

    TextureSpecification spec;
    spec.width                         = 1;
    spec.height                        = 1;
    s_pathtracer->impl->output_texture = atcg::Texture2D::create(spec);

    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
    CUcontext cuCtx                   = 0;

    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &s_pathtracer->impl->context));
}

void Pathtracer::destroy()
{
    s_pathtracer->impl->stop();
    delete s_pathtracer;
}

void Pathtracer::Impl::start()
{
    if(!running)
    {
        running       = true;
        worker_thread = std::thread(&Pathtracer::Impl::worker, this);
    }
}

void Pathtracer::Impl::stop()
{
    running = false;

    if(worker_thread.joinable()) worker_thread.join();
}

void Pathtracer::Impl::resize(const uint32_t width, const uint32_t height)
{
    // Resize
    s_pathtracer->impl->swap_chain_buffer =
        torch::zeros({2, height, width, 4}, atcg::TensorOptions::uint8DeviceOptions());

    s_pathtracer->impl->output_buffer = (glm::u8vec4*)s_pathtracer->impl->swap_chain_buffer.index({0}).data_ptr();

    TextureSpecification spec;
    spec.width                         = width;
    spec.height                        = height;
    s_pathtracer->impl->output_texture = atcg::Texture2D::create(spec);
}

atcg::ref_ptr<Texture2D> Pathtracer::getOutputTexture()
{
    {
        std::lock_guard guard(s_pathtracer->impl->swap_chain_mutex);

        if(s_pathtracer->impl->dirty)
        {
            s_pathtracer->impl->output_texture->setData(
                s_pathtracer->impl->swap_chain_buffer.index({s_pathtracer->impl->swap_index}));
            s_pathtracer->impl->dirty = false;
        }
    }

    return s_pathtracer->impl->output_texture;
}

void Pathtracer::draw(const atcg::ref_ptr<Scene>& scene,
                      const atcg::ref_ptr<PerspectiveCamera>& camera,
                      const atcg::ref_ptr<RaytracingShader>& shader,
                      uint32_t width,
                      uint32_t height,
                      const uint32_t num_samples)
{
    s_pathtracer->impl->raytracing_pipeline = atcg::make_ref<RayTracingPipeline>(s_pathtracer->impl->context);
    s_pathtracer->impl->sbt                 = atcg::make_ref<ShaderBindingTable>(s_pathtracer->impl->context);

    s_pathtracer->impl->resize(width, height);

    // TODO: Make this the passed shader
    s_pathtracer->impl->shader = atcg::make_ref<PathtracingShader>(s_pathtracer->impl->context, scene, camera);
    s_pathtracer->impl->shader->initializePipeline(s_pathtracer->impl->raytracing_pipeline, s_pathtracer->impl->sbt);
    s_pathtracer->impl->shader->setCamera(camera);

    s_pathtracer->impl->raytracing_pipeline->createPipeline();
    s_pathtracer->impl->sbt->createSBT();

    s_pathtracer->impl->num_samples = num_samples;

    s_pathtracer->impl->start();
}

}    // namespace atcg