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

    void worker();

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
};

void Pathtracer::Impl::worker()
{
    while(true)
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

    s_pathtracer->impl->raytracing_pipeline = atcg::make_ref<RayTracingPipeline>(s_pathtracer->impl->context);
    s_pathtracer->impl->sbt                 = atcg::make_ref<ShaderBindingTable>(s_pathtracer->impl->context);
}

void Pathtracer::bakeScene(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<PerspectiveCamera>& camera)
{
    s_pathtracer->impl->shader = atcg::make_ref<PathtracingShader>(s_pathtracer->impl->context, scene, camera);
    s_pathtracer->impl->shader->initializePipeline(s_pathtracer->impl->raytracing_pipeline, s_pathtracer->impl->sbt);

    s_pathtracer->impl->raytracing_pipeline->createPipeline();
    s_pathtracer->impl->sbt->createSBT();
}

void Pathtracer::start()
{
    if(!s_pathtracer->impl->running)
    {
        s_pathtracer->impl->running       = true;
        s_pathtracer->impl->worker_thread = std::thread(&Pathtracer::Impl::worker, s_pathtracer->impl.get());
    }
}

void Pathtracer::stop()
{
    if(s_pathtracer->impl->running)
    {
        s_pathtracer->impl->running = false;

        if(s_pathtracer->impl->worker_thread.joinable()) s_pathtracer->impl->worker_thread.join();
    }
}

void Pathtracer::reset(const atcg::ref_ptr<PerspectiveCamera>& camera)
{
    stop();

    if(s_pathtracer->impl->shader)
    {
        s_pathtracer->impl->shader->reset();
        s_pathtracer->impl->shader->setCamera(camera);
    }

    start();
}

void Pathtracer::resize(const uint32_t width, const uint32_t height)
{
    bool running = s_pathtracer->impl->running;
    stop();

    // Resize
    s_pathtracer->impl->swap_chain_buffer =
        torch::zeros({2, height, width, 4}, atcg::TensorOptions::uint8DeviceOptions());

    s_pathtracer->impl->output_buffer = (glm::u8vec4*)s_pathtracer->impl->swap_chain_buffer.index({0}).data_ptr();

    TextureSpecification spec;
    spec.width                         = width;
    spec.height                        = height;
    s_pathtracer->impl->output_texture = atcg::Texture2D::create(spec);

    if(s_pathtracer->impl->shader) s_pathtracer->impl->shader->reset();

    if(running) start();
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

}    // namespace atcg