#include <Renderer/Pathtracer.h>

#include <Scene/Components.h>
#include <Math/Tracing.h>
#include <DataStructure/Timer.h>
#include <Math/Random.h>
#include <Renderer/BSDFModels.h>

#include <thread>
#include <mutex>
#include <execution>

namespace atcg
{

Pathtracer* Pathtracer::s_pathtracer = new Pathtracer;

class Pathtracer::Impl
{
public:
    Impl() {}
    ~Impl() {}

    void worker();


    // Basic swap chain
    uint32_t width  = 512;
    uint32_t height = 512;
    glm::u8vec4* output_buffer;
    uint8_t swap_index = 1;
    atcg::dref_ptr<glm::u8vec4> swap_chain_buffer;
    bool dirty = false;
    std::mutex swap_chain_mutex;

    atcg::ref_ptr<atcg::Texture2D> output_texture;

    std::thread worker_thread;

    std::atomic_bool running = false;
};

void Pathtracer::Impl::worker()
{
    uint32_t frame_counter = 0;
    while(true)
    {
        Timer frame_time;

        // TODO: Pixel shader

        // Perform swap
        {
            std::lock_guard guard(swap_chain_mutex);
            output_buffer = swap_chain_buffer.get() + swap_index * width * height;
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

    s_pathtracer->impl->swap_chain_buffer = atcg::dref_ptr<glm::u8vec4>(
        2 * s_pathtracer->impl->width * s_pathtracer->impl->height);    // 2 swap chain buffers
    s_pathtracer->impl->output_buffer = s_pathtracer->impl->swap_chain_buffer.get();

    TextureSpecification spec;
    spec.width                         = s_pathtracer->impl->width;
    spec.height                        = s_pathtracer->impl->height;
    s_pathtracer->impl->output_texture = atcg::Texture2D::create(spec);
}

void Pathtracer::bakeScene(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<PerspectiveCamera>& camera) {}

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
    s_pathtracer->impl->running = false;

    if(s_pathtracer->impl->worker_thread.joinable()) s_pathtracer->impl->worker_thread.join();
}

atcg::ref_ptr<Texture2D> Pathtracer::getOutputTexture()
{
    {
        std::lock_guard guard(s_pathtracer->impl->swap_chain_mutex);

        if(s_pathtracer->impl->dirty)
        {
            glm::u8vec4* data_ptr = s_pathtracer->impl->swap_chain_buffer.get() + s_pathtracer->impl->swap_index *
                                                                                      s_pathtracer->impl->width *
                                                                                      s_pathtracer->impl->height;
            // s_pathtracer->impl->output_texture->setData((void*)data_ptr);
            s_pathtracer->impl->dirty = false;
        }
    }

    return s_pathtracer->impl->output_texture;
}

}    // namespace atcg