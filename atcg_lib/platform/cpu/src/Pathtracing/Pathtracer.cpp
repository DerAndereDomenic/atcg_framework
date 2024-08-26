#include <Pathtracing/Pathtracer.h>

#include <Scene/Components.h>
#include <Pathtracing/Tracing.h>
#include <DataStructure/Timer.h>
#include <Math/Random.h>
#include <Pathtracing/BSDF.h>
#include <Scene/Entity.h>
#include <Pathtracing/RaytracingShader.h>
#include <Pathtracing/RaytracingShaderManager.h>

#include <Pathtracing/PathtracingShader.h>

#include <thread>
#include <mutex>
#include <execution>

namespace atcg
{
class PathtracingSystem::Impl
{
public:
    Impl() {}
    ~Impl() {}

    void start();
    void worker();
    void resize(const uint32_t width, const uint32_t height);

    void draw(const atcg::ref_ptr<Scene>& scene,
              const atcg::ref_ptr<PerspectiveCamera>& camera,
              const atcg::ref_ptr<RaytracingShader>& shader,
              uint32_t width,
              uint32_t height);

    float rendering_time = 0.0f;
    uint32_t subframe    = 0;

    atcg::ref_ptr<RaytracingShader> shader = nullptr;
    atcg::ref_ptr<atcg::Texture2D> output_texture;

    std::thread worker_thread;
    glm::u8vec4* output_buffer;
    uint8_t swap_index = 1;
    torch::Tensor swap_chain_buffer;
    bool dirty = false;
    std::mutex swap_chain_mutex;
    std::atomic_bool running = false;

    uint32_t max_num_samples;
    float max_rendering_time;
    bool samples_mode = true;

    uint32_t width = 0, height = 0;

    atcg::ref_ptr<RayTracingPipeline> pipeline;
    atcg::ref_ptr<ShaderBindingTable> sbt;
};

void PathtracingSystem::Impl::worker()
{
    rendering_time = 0.0f;
    subframe       = 0;

    Timer render_timer;
    while(true)
    {
        Timer frame_time;

        shader->generateRays(
            atcg::createDeviceTensorFromPointer<uint8_t>((uint8_t*)output_buffer,
                                                         {output_texture->height(), output_texture->width(), 4}));

        // Perform swap
        {
            std::lock_guard guard(swap_chain_mutex);
            output_buffer = (glm::u8vec4*)swap_chain_buffer.index({swap_index}).data_ptr();
            swap_index    = (swap_index + 1) % 2;
            dirty         = true;
        }

        ++subframe;
        rendering_time += frame_time.elapsedSeconds();

        if(samples_mode && subframe >= max_num_samples) break;
        if(!samples_mode && rendering_time >= max_rendering_time) break;

        if(!running) break;
    }

    running = false;
}

PathtracingSystem::PathtracingSystem() {}

PathtracingSystem::~PathtracingSystem()
{
    stop();
}

void PathtracingSystem::init()
{
    impl = std::make_unique<Impl>();

    impl->resize(1, 1);

    {
        auto shader = atcg::make_ref<PathtracingShader>();
        atcg::RaytracingShaderManager::addShader("Pathtracing", shader);
    }
}

void PathtracingSystem::Impl::start()
{
    if(!running)
    {
        running = true;
        if(worker_thread.joinable()) worker_thread.join();
        worker_thread = std::thread(&PathtracingSystem::Impl::worker, this);
    }
}

void PathtracingSystem::stop()
{
    impl->running = false;

    if(impl->worker_thread.joinable()) impl->worker_thread.join();
}

bool PathtracingSystem::isRunning() const
{
    return impl->running;
}

void PathtracingSystem::Impl::resize(const uint32_t width, const uint32_t height)
{
    // Resize
    swap_chain_buffer = torch::zeros({2, height, width, 4}, atcg::TensorOptions::uint8DeviceOptions());

    output_buffer = (glm::u8vec4*)swap_chain_buffer.index({0}).data_ptr();

    TextureSpecification spec;
    spec.width     = width;
    spec.height    = height;
    output_texture = atcg::Texture2D::create(spec);

    this->width  = width;
    this->height = height;
}

void PathtracingSystem::Impl::draw(const atcg::ref_ptr<Scene>& scene,
                                   const atcg::ref_ptr<PerspectiveCamera>& camera,
                                   const atcg::ref_ptr<RaytracingShader>& pshader,
                                   uint32_t width,
                                   uint32_t height)
{
    swap_index = 1;

    pipeline = atcg::make_ref<RayTracingPipeline>(nullptr);
    sbt      = atcg::make_ref<ShaderBindingTable>(nullptr);

    resize(width, height);

    shader = std::dynamic_pointer_cast<RaytracingShader>(pshader);
    shader->setScene(scene);
    shader->setCamera(camera);
    shader->initializePipeline(pipeline, sbt);

    start();
}


atcg::ref_ptr<Texture2D> PathtracingSystem::getOutputTexture()
{
    {
        std::lock_guard guard(impl->swap_chain_mutex);

        if(impl->dirty)
        {
            impl->output_texture->setData(impl->swap_chain_buffer.index({impl->swap_index}));
            impl->dirty = false;
        }
    }

    return impl->output_texture;
}

void PathtracingSystem::draw(const atcg::ref_ptr<Scene>& scene,
                             const atcg::ref_ptr<PerspectiveCamera>& camera,
                             const atcg::ref_ptr<RaytracingShader>& shader,
                             uint32_t width,
                             uint32_t height,
                             const uint32_t num_samples)
{
    impl->samples_mode    = true;
    impl->max_num_samples = num_samples;

    impl->draw(scene, camera, shader, width, height);
}

void PathtracingSystem::draw(const atcg::ref_ptr<Scene>& scene,
                             const atcg::ref_ptr<PerspectiveCamera>& camera,
                             const atcg::ref_ptr<RaytracingShader>& shader,
                             uint32_t width,
                             uint32_t height,
                             const float time)
{
    impl->samples_mode       = false;
    impl->max_rendering_time = time;

    impl->draw(scene, camera, shader, width, height);
}

uint32_t PathtracingSystem::getFrameIndex() const
{
    return impl->subframe;
}

uint32_t PathtracingSystem::getWidth() const
{
    return impl->width;
}

uint32_t PathtracingSystem::getHeight() const
{
    return impl->height;
}

float PathtracingSystem::getLastRenderingTime() const
{
    return impl->rendering_time;
}

uint32_t PathtracingSystem::getSampleCount() const
{
    return impl->subframe;
}

// class Pathtracer::Impl
// {
// public:
//     Impl() {}
//     ~Impl() {}

//     void worker();

//     // Baked scene

//     std::vector<glm::vec3> positions;
//     std::vector<glm::vec3> normals;
//     std::vector<glm::vec3> uvs;
//     std::vector<glm::u32vec3> faces;
//     std::vector<uint32_t> mesh_idx;
//     nanort::BVHAccel<float> accel;

//     std::vector<atcg::ref_ptr<Image>> diffuse_images;
//     std::vector<atcg::ref_ptr<Image>> roughness_images;
//     std::vector<atcg::ref_ptr<Image>> metallic_images;

//     atcg::ref_ptr<PerspectiveCamera> camera;
//     bool hasSkybox = false;
//     atcg::ref_ptr<Image> skybox_image;
//     glm::vec3 read_image(const atcg::ref_ptr<Image>& image, const glm::vec2& uv);

//     std::vector<glm::vec3> accumulation_buffer;

//     // Basic swap chain
//     uint32_t width  = 512;
//     uint32_t height = 512;
//     glm::u8vec4* output_buffer;
//     uint8_t swap_index = 1;
//     atcg::DeviceBuffer<glm::u8vec4> swap_chain_buffer;
//     bool dirty = false;
//     std::mutex swap_chain_mutex;
//     std::vector<uint32_t> horizontalScanLine;
//     std::vector<uint32_t> verticalScanLine;

//     atcg::ref_ptr<atcg::Texture2D> output_texture;

//     std::thread worker_thread;

//     std::atomic_bool running = false;
// };

// glm::vec3 Pathtracer::Impl::read_image(const atcg::ref_ptr<Image>& image, const glm::vec2& uv)
// {
//     glm::vec4 result(0);

//     glm::ivec2 pixel(uv.x * image->width(), image->height() - uv.y * image->height());
//     pixel = glm::clamp(pixel, glm::ivec2(0), glm::ivec2(image->width() - 1, image->height() - 1));

//     glm::vec3 color;

//     uint32_t channels = image->channels();
//     if(image->isHDR())
//     {
//         if(channels == 4)
//         {
//             color = image->dataAt<glm::vec4>(pixel.x + image->width() * pixel.y);
//         }
//         else
//         {
//             color = glm::vec3(image->dataAt<float>(pixel.x + image->width() * pixel.y));
//         }
//     }
//     else
//     {
//         if(channels == 4)
//         {
//             glm::u8vec4 val = image->dataAt<glm::u8vec4>(pixel.x + image->width() * pixel.y);
//             color           = glm::vec3((float)val.x, (float)val.y, (float)val.z) / 255.0f;
//         }
//         else
//         {
//             uint8_t val = image->dataAt<uint8_t>(pixel.x + image->width() * pixel.y);
//             color       = glm::vec3((float)val) / 255.0f;
//         }
//     }

//     return color;
// }

// void Pathtracer::Impl::worker()
// {
//     uint32_t frame_counter = 0;
//     while(true)
//     {
//         Timer frame_time;

//         glm::mat4 camera_view = glm::inverse(camera->getView());
//         glm::vec3 cam_eye     = camera_view[3];
//         glm::vec3 U           = glm::normalize(camera_view[0]);
//         glm::vec3 V           = glm::normalize(camera_view[1]);
//         glm::vec3 W           = -glm::normalize(camera_view[2]);

//         int max_trace_depth = 10;

//         std::for_each(std::execution::par,
//                       verticalScanLine.begin(),
//                       verticalScanLine.end(),
//                       [&](uint32_t y)
//                       {
//                           std::for_each(
//                               std::execution::par,
//                               horizontalScanLine.begin(),
//                               horizontalScanLine.end(),
//                               [&](uint32_t x)
//                               {
//                                   uint64_t seed = sampleTEA64(x + width * y, frame_counter);
//                                   atcg::PCG32 rng(seed);

//                                   glm::vec2 jitter = rng.next2d();
//                                   float u          = (((float)x + jitter.x) / (float)width - 0.5f) * 2.0f;
//                                   float v          = (((float)(height - y) + jitter.y) / (float)height - 0.5f)
//                                   * 2.0f;

//                                   glm::vec3 ray_dir    = glm::normalize(u * U + v * V + W);
//                                   glm::vec3 ray_origin = cam_eye;
//                                   glm::vec3 radiance(0);
//                                   glm::vec3 throughput(1);

//                                   glm::vec3 next_origin;
//                                   glm::vec3 next_dir;

//                                   for(int n = 0; n < max_trace_depth; ++n)
//                                   {
//                                       SurfaceInteraction si = Tracing::traceRay(accel,
//                                                                                 positions,
//                                                                                 normals,
//                                                                                 uvs,
//                                                                                 faces,
//                                                                                 ray_origin,
//                                                                                 ray_dir,
//                                                                                 1e-3f,
//                                                                                 1e6f);
//                                       if(si.valid)
//                                       {
//                                           // PBR Sampling
//                                           uint32_t mesh_index     = mesh_idx[si.primitive_idx];
//                                           glm::vec3 diffuse_color = read_image(diffuse_images[mesh_index], si.uv);
//                                           float metallic          = read_image(metallic_images[mesh_index], si.uv).x;
//                                           float roughness         = read_image(roughness_images[mesh_index],
//                                           si.uv).x;

//                                           glm::vec3 metallic_color =
//                                               (1.0f - metallic) * glm::vec3(0.04) + metallic * diffuse_color;

//                                           BSDFSamplingResult result =
//                                               sampleGGX(si, diffuse_color, metallic_color, metallic, roughness, rng);
//                                           if(result.sample_probability > 0.0f)
//                                           {
//                                               next_origin = si.position;
//                                               next_dir    = result.out_dir;
//                                               throughput *= result.bsdf_weight;
//                                           }
//                                           else
//                                           {
//                                               break;
//                                           }
//                                       }
//                                       else
//                                       {
//                                           float theta = std::acos(ray_dir.y) / glm::pi<float>();
//                                           float phi   = (std::atan2(ray_dir.z, ray_dir.x) + glm::pi<float>()) /
//                                                       (2.0f * glm::pi<float>());

//                                           if(hasSkybox)
//                                               radiance = throughput * read_image(skybox_image, glm::vec2(phi,
//                                               theta));

//                                           break;
//                                       }

//                                       ray_origin = next_origin;
//                                       ray_dir    = next_dir;
//                                   }

//                                   if(frame_counter > 0)
//                                   {
//                                       // Mix with previous subframes if present!
//                                       const float a = 1.0f / static_cast<float>(frame_counter + 1);
//                                       const glm::vec3 prev_output_radiance = accumulation_buffer[x + width * y];
//                                       radiance = glm::lerp(prev_output_radiance, radiance, a);
//                                   }

//                                   accumulation_buffer[x + width * y] = radiance;

//                                   glm::vec3 tone_mapped =
//                                       glm::clamp(glm::pow(1.0f - glm::exp(-radiance), glm::vec3(1.0f / 2.2f)),
//                                                  glm::vec3(0),
//                                                  glm::vec3(1));

//                                   output_buffer[x + width * y] = glm::vec4((uint8_t)(tone_mapped.x * 255.0f),
//                                                                            (uint8_t)(tone_mapped.y * 255.0f),
//                                                                            (uint8_t)(tone_mapped.z * 255.0f),
//                                                                            255);
//                               });
//                       });

//         ++frame_counter;
//         std::cout << frame_time.elapsedMillis() << "\n";

//         // Perform swap
//         {
//             std::lock_guard guard(swap_chain_mutex);
//             output_buffer = swap_chain_buffer.get() + swap_index * width * height;
//             swap_index    = (swap_index + 1) % 2;
//             dirty         = true;
//         }

//         if(!running) break;
//     }
// }

// Pathtracer::Pathtracer() {}

// Pathtracer::~Pathtracer() {}

// void Pathtracer::init()
// {
//     impl = std::make_unique<Impl>();

//     impl->swap_chain_buffer = atcg::DeviceBuffer<glm::u8vec4>(
//         2 * impl->width * impl->height);    // 2 swap chain buffers
//     impl->output_buffer = impl->swap_chain_buffer.get();

//     impl->accumulation_buffer.resize(impl->width * impl->height);
//     memset(impl->accumulation_buffer.data(),
//            0,
//            sizeof(glm::vec3) * impl->accumulation_buffer.size());

//     TextureSpecification spec;
//     spec.width                         = impl->width;
//     spec.height                        = impl->height;
//     impl->output_texture = atcg::Texture2D::create(spec);

//     impl->horizontalScanLine.resize(impl->width);
//     impl->verticalScanLine.resize(impl->height);
//     for(int i = 0; i < impl->width; ++i)
//     {
//         impl->horizontalScanLine[i] = i;
//     }

//     for(int i = 0; i < impl->height; ++i)
//     {
//         impl->verticalScanLine[i] = i;
//     }
// }

// void Pathtracer::bakeScene(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<PerspectiveCamera>& camera)
// {
//     impl->diffuse_images.clear();
//     impl->roughness_images.clear();
//     impl->metallic_images.clear();
//     impl->positions.clear();
//     impl->normals.clear();
//     impl->uvs.clear();
//     impl->faces.clear();
//     impl->mesh_idx.clear();

//     impl->camera = camera;

//     auto view = scene->getAllEntitiesWith<GeometryComponent, MeshRenderComponent, TransformComponent>();

//     uint32_t total_vertices = 0;
//     uint32_t total_faces    = 0;
//     std::vector<atcg::ref_ptr<Graph>> geometry;
//     std::vector<TransformComponent> transforms;
//     for(auto e: view)
//     {
//         Entity entity(e, scene.get());

//         TransformComponent transform = entity.getComponent<TransformComponent>();

//         transforms.push_back(transform);

//         atcg::ref_ptr<Graph> graph = entity.getComponent<GeometryComponent>().graph;
//         geometry.push_back(graph);

//         total_vertices += graph->n_vertices();
//         total_faces += graph->n_faces();

//         auto& material = entity.getComponent<MeshRenderComponent>().material;

//         impl->diffuse_images.push_back(atcg::make_ref<Image>(material.getDiffuseTexture()));
//         impl->roughness_images.push_back(atcg::make_ref<Image>(material.getRoughnessTexture()));
//         impl->metallic_images.push_back(atcg::make_ref<Image>(material.getMetallicTexture()));
//     }

//     impl->positions.resize(total_vertices);
//     impl->normals.resize(total_vertices);
//     impl->uvs.resize(total_vertices);
//     impl->faces.resize(total_faces);
//     impl->mesh_idx.resize(total_faces);

//     uint32_t vertex_idx = 0;
//     uint32_t face_idx   = 0;

//     for(int i = 0; i < geometry.size(); ++i)
//     {
//         auto& graph      = geometry[i];
//         Vertex* vertices = graph->getVerticesBuffer()->getHostPointer<Vertex>();

//         TransformComponent& transform = transforms[i];
//         glm::mat4 model               = transform.getModel();
//         glm::mat4 normal_matrix       = glm::transpose(glm::inverse(model));

//         glm::u32vec3* faces = graph->getFaceIndexBuffer()->getHostPointer<glm::u32vec3>();

//         for(int j = 0; j < graph->n_faces(); ++j)
//         {
//             impl->faces[face_idx]    = faces[j] + vertex_idx;
//             impl->mesh_idx[face_idx] = i;
//             ++face_idx;
//         }

//         for(int j = 0; j < graph->n_vertices(); ++j)
//         {
//             impl->positions[vertex_idx] = model * glm::vec4(vertices[j].position, 1.0f);
//             impl->normals[vertex_idx] =
//                 glm::normalize(glm::vec3(normal_matrix * glm::vec4(vertices[j].normal, 0.0f)));
//             impl->uvs[vertex_idx] = vertices[j].uv;
//             ++vertex_idx;
//         }

//         graph->getVerticesBuffer()->unmapHostPointers();
//         graph->getFaceIndexBuffer()->unmapHostPointers();
//     }

//     nanort::TriangleMesh<float> triangle_mesh(reinterpret_cast<const float*>(impl->positions.data()),
//                                               reinterpret_cast<const uint32_t*>(impl->faces.data()),
//                                               sizeof(float) * 3);
//     nanort::TriangleSAHPred<float> triangle_pred(reinterpret_cast<const
//     float*>(impl->positions.data()),
//                                                  reinterpret_cast<const uint32_t*>(impl->faces.data()),
//                                                  sizeof(float) * 3);
//     bool ret = impl->accel.Build(total_faces, triangle_mesh, triangle_pred);
//     assert(ret);

//     nanort::BVHBuildStatistics stats = impl->accel.GetStatistics();

//     ATCG_INFO("BVH statistics:");
//     ATCG_INFO("\t# of leaf   nodes: {0}", stats.num_leaf_nodes);
//     ATCG_INFO("\t# of branch nodes: {0}", stats.num_branch_nodes);
//     ATCG_INFO("\tMax tree depth   : {0}", stats.max_tree_depth);

//     impl->hasSkybox = Renderer::hasSkybox();
//     if(impl->hasSkybox)
//     {
//         atcg::ref_ptr<Texture2D> skybox_texture = atcg::Renderer::getSkyboxTexture();

//         impl->skybox_image = atcg::make_ref<Image>(skybox_texture);
//     }
// }

// void Pathtracer::start()
// {
//     if(!impl->running)
//     {
//         memset(impl->accumulation_buffer.data(),
//                0,
//                sizeof(glm::vec3) * impl->accumulation_buffer.size());
//         impl->running       = true;
//         impl->worker_thread = std::thread(&Pathtracer::Impl::worker, impl.get());
//     }
// }

// void Pathtracer::stop()
// {
//     impl->running = false;

//     if(impl->worker_thread.joinable()) impl->worker_thread.join();
// }

// atcg::ref_ptr<Texture2D> Pathtracer::getOutputTexture()
// {
//     {
//         std::lock_guard guard(impl->swap_chain_mutex);

//         if(impl->dirty)
//         {
//             glm::u8vec4* data_ptr = impl->swap_chain_buffer.get() + impl->swap_index *
//                                                                                       impl->width *
//                                                                                       impl->height;
//             impl->output_texture->setData((void*)data_ptr);
//             impl->dirty = false;
//         }
//     }

//     return impl->output_texture;
// }

}    // namespace atcg