#include <Renderer/Pathtracer.h>

#include <Scene/Components.h>
#include <Math/Tracing.h>
#include <DataStructure/Timer.h>
#include <Math/Random.h>
#include <Renderer/BSDFModels.h>

#include <Scene/Entity.h>
#include "Params.cuh"

#include <thread>
#include <mutex>
#include <execution>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "Common.h"
#include "RaytracingPipeline.h"
#include "ShaderBindingTable.h"

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

    atcg::dref_ptr<Params> launch_params;

    // Baked scene
    std::vector<torch::Tensor> vertices;
    std::vector<torch::Tensor> normals;
    std::vector<torch::Tensor> uvs;
    std::vector<torch::Tensor> faces;
    std::vector<atcg::DeviceBuffer<HitGroupData>> hit_data;
    std::vector<atcg::DeviceBuffer<uint8_t>> gas_buffers;
    atcg::DeviceBuffer<uint8_t> ias_buffer;
    std::vector<OptixTraversableHandle> gas_handles;
    OptixTraversableHandle ias_handle;
    atcg::ref_ptr<PerspectiveCamera> camera;
    bool hasSkybox = false;
    atcg::DeviceBuffer<glm::vec3> skybox_data;
    uint32_t skybox_width;
    uint32_t skybox_height;

    atcg::DeviceBuffer<glm::vec3> accumulation_buffer;

    // Basic swap chain
    uint32_t width  = 512;
    uint32_t height = 512;
    glm::u8vec4* output_buffer;
    uint8_t swap_index = 1;
    atcg::DeviceBuffer<glm::u8vec4> swap_chain_buffer;
    bool dirty = false;
    std::mutex swap_chain_mutex;

    atcg::ref_ptr<atcg::Texture2D> output_texture;

    std::thread worker_thread;

    std::atomic_bool running = false;

    uint32_t raygen_index = 0;
};

void Pathtracer::Impl::worker()
{
    uint32_t frame_counter = 0;
    while(true)
    {
        Timer frame_time;

        Params params;

        glm::mat4 camera_view = glm::inverse(camera->getView());
        memcpy(params.cam_eye, glm::value_ptr(camera_view[3]), sizeof(glm::vec3));
        memcpy(params.U, glm::value_ptr(glm::normalize(camera_view[0])), sizeof(glm::vec3));
        memcpy(params.V, glm::value_ptr(glm::normalize(camera_view[1])), sizeof(glm::vec3));
        memcpy(params.W, glm::value_ptr(-glm::normalize(camera_view[2])), sizeof(glm::vec3));

        params.accumulation_buffer = accumulation_buffer.get();
        params.output_image        = output_buffer;
        params.image_height        = width;
        params.image_width         = height;
        params.handle              = ias_handle;
        params.frame_counter       = frame_counter;

        params.hasSkybox     = hasSkybox;
        params.skybox_width  = skybox_width;
        params.skybox_height = skybox_height;
        params.skybox_data   = skybox_data.get();

        launch_params.upload(&params);

        OPTIX_CHECK(optixLaunch(raytracing_pipeline->getPipeline(),
                                0,    // Default CUDA stream
                                (CUdeviceptr)launch_params.get(),
                                sizeof(Params),
                                sbt->getSBT(raygen_index),
                                width,
                                height,
                                1));    // depth

        SYNCHRONIZE_DEFAULT_STREAM();

        ATCG_TRACE(frame_time.elapsedMillis());

        ++frame_counter;

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

    s_pathtracer->impl->swap_chain_buffer = atcg::DeviceBuffer<glm::u8vec4>(
        2 * s_pathtracer->impl->width * s_pathtracer->impl->height);    // 2 swap chain buffers
    s_pathtracer->impl->output_buffer = s_pathtracer->impl->swap_chain_buffer.get();

    s_pathtracer->impl->accumulation_buffer =
        atcg::DeviceBuffer<glm::vec3>(s_pathtracer->impl->width * s_pathtracer->impl->height);

    TextureSpecification spec;
    spec.width                         = s_pathtracer->impl->width;
    spec.height                        = s_pathtracer->impl->height;
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
    s_pathtracer->impl->camera = camera;

    if(Renderer::hasSkybox())
    {
        s_pathtracer->impl->hasSkybox     = true;
        auto skybox_texture               = Renderer::getSkyboxTexture();
        s_pathtracer->impl->skybox_width  = skybox_texture->width();
        s_pathtracer->impl->skybox_height = skybox_texture->height();

        auto data = skybox_texture->getData(atcg::CPU);

        s_pathtracer->impl->skybox_data =
            atcg::DeviceBuffer<glm::vec3>(s_pathtracer->impl->skybox_width * s_pathtracer->impl->skybox_height);
        s_pathtracer->impl->skybox_data.upload((glm::vec3*)data.data_ptr());
    }

    // Extract scene information
    auto view = scene->getAllEntitiesWith<GeometryComponent, MeshRenderComponent, TransformComponent>();
    std::vector<TransformComponent> transforms;
    for(auto e: view)
    {
        Entity entity(e, scene.get());

        TransformComponent transform = entity.getComponent<TransformComponent>();

        transforms.push_back(transform);

        atcg::ref_ptr<Graph> graph = entity.getComponent<GeometryComponent>().graph;

        auto d_vertices = graph->getDevicePositions().clone();
        auto d_normals  = graph->getDeviceNormals().clone();
        auto d_uvs      = graph->getDeviceUVs().clone();
        auto d_faces    = graph->getDeviceFaces().clone();

        s_pathtracer->impl->vertices.push_back(d_vertices);
        s_pathtracer->impl->normals.push_back(d_normals);
        s_pathtracer->impl->uvs.push_back(d_uvs);
        s_pathtracer->impl->faces.push_back(d_faces);

        graph->unmapAllPointers();
    }

    s_pathtracer->impl->gas_handles.resize(s_pathtracer->impl->vertices.size());

    // Build Accel structure

    for(int i = 0; i < s_pathtracer->impl->vertices.size(); ++i)
    {
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags             = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

        const uint32_t triangle_input_flags[1]     = {OPTIX_GEOMETRY_FLAG_NONE};
        OptixBuildInput triangle_input             = {};
        triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.numVertices   = s_pathtracer->impl->vertices[i].numel();
        CUdeviceptr ptr                            = (CUdeviceptr)s_pathtracer->impl->vertices[i].data_ptr();
        triangle_input.triangleArray.vertexBuffers = &ptr;
        triangle_input.triangleArray.flags         = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;

        triangle_input.triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input.triangleArray.numIndexTriplets = s_pathtracer->impl->faces[i].size(0);
        triangle_input.triangleArray.indexBuffer      = (CUdeviceptr)s_pathtracer->impl->faces[i].data_ptr();

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(s_pathtracer->impl->context,
                                                 &accel_options,
                                                 &triangle_input,
                                                 1,
                                                 &gas_buffer_sizes));

        atcg::DeviceBuffer<uint8_t> d_temp_buffer_gas(gas_buffer_sizes.tempSizeInBytes);
        auto gas_buffer = atcg::DeviceBuffer<uint8_t>(gas_buffer_sizes.outputSizeInBytes);
        s_pathtracer->impl->gas_buffers.push_back(gas_buffer);

        OPTIX_CHECK(optixAccelBuild(s_pathtracer->impl->context,
                                    0,    // CUDA stream
                                    &accel_options,
                                    &triangle_input,
                                    1,    // num build inputs
                                    (CUdeviceptr)d_temp_buffer_gas.get(),
                                    gas_buffer_sizes.tempSizeInBytes,
                                    (CUdeviceptr)gas_buffer.get(),
                                    gas_buffer_sizes.outputSizeInBytes,
                                    &s_pathtracer->impl->gas_handles[i],    // Output handle to the struct
                                    nullptr,                                // emitted property list
                                    0));                                    // num emitted properties
    }

    // IAS

    const size_t num_instances = s_pathtracer->impl->vertices.size();

    std::vector<OptixInstance> optix_instances(num_instances);
    for(size_t i = 0; i < num_instances; ++i)
    {
        auto& optix_instance = optix_instances[i];
        memset(&optix_instance, 0, sizeof(OptixInstance));

        optix_instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        optix_instance.instanceId        = i;
        optix_instance.sbtOffset         = i;
        optix_instance.visibilityMask    = 1;
        optix_instance.traversableHandle = s_pathtracer->impl->gas_handles[i];

        reinterpret_cast<glm::mat3x4&>(optix_instance.transform) =
            glm::transpose(glm::mat4x3(transforms[i].getModel()));
    }

    atcg::DeviceBuffer<OptixInstance> d_instances(num_instances);
    d_instances.upload(optix_instances.data());

    OptixBuildInput instance_input            = {};
    instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances    = (CUdeviceptr)d_instances.get();
    instance_input.instanceArray.numInstances = static_cast<uint32_t>(num_instances);

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(s_pathtracer->impl->context,
                                             &accel_options,
                                             &instance_input,
                                             1,    // num build inputs
                                             &ias_buffer_sizes));

    atcg::DeviceBuffer<uint8_t> d_temp_buffer(ias_buffer_sizes.tempSizeInBytes);
    s_pathtracer->impl->ias_buffer = atcg::DeviceBuffer<uint8_t>(ias_buffer_sizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(s_pathtracer->impl->context,
                                nullptr,    // CUDA stream
                                &accel_options,
                                &instance_input,
                                1,    // num build inputs
                                (CUdeviceptr)d_temp_buffer.get(),
                                ias_buffer_sizes.tempSizeInBytes,
                                (CUdeviceptr)s_pathtracer->impl->ias_buffer.get(),
                                ias_buffer_sizes.outputSizeInBytes,
                                &s_pathtracer->impl->ias_handle,
                                nullptr,    // emitted property list
                                0           // num emitted properties
                                ));


    const std::string ptx_filename = "C:/Users/Domenic/Documents/Repositories/atcg_framework/build/ptxmodules.dir/"
                                     "Debug/raygen.ptx";
    OptixProgramGroup raygen_prog_group =
        s_pathtracer->impl->raytracing_pipeline->addRaygenShader({ptx_filename, "__raygen__rg"});
    OptixProgramGroup miss_prog_group =
        s_pathtracer->impl->raytracing_pipeline->addMissShader({ptx_filename, "__miss__ms"});
    OptixProgramGroup hitgroup_prog_group =
        s_pathtracer->impl->raytracing_pipeline->addTrianglesHitGroupShader({ptx_filename, "__closesthit__ch"}, {});

    s_pathtracer->impl->raytracing_pipeline->createPipeline();

    // SBT
    s_pathtracer->impl->raygen_index = s_pathtracer->impl->sbt->addRaygenEntry(raygen_prog_group);
    s_pathtracer->impl->sbt->addMissEntry(miss_prog_group);

    for(int i = 0; i < s_pathtracer->impl->vertices.size(); ++i)
    {
        HitGroupData hit_data;
        hit_data.positions = (glm::vec3*)s_pathtracer->impl->vertices[i].data_ptr();
        hit_data.normals   = (glm::vec3*)s_pathtracer->impl->normals[i].data_ptr();
        hit_data.uvs       = (glm::vec3*)s_pathtracer->impl->uvs[i].data_ptr();
        hit_data.faces     = (glm::u32vec3*)s_pathtracer->impl->faces[i].data_ptr();

        DeviceBuffer<HitGroupData> d_hit_data(1);
        d_hit_data.upload(&hit_data);
        s_pathtracer->impl->hit_data.push_back(d_hit_data);

        s_pathtracer->impl->sbt->addHitEntry(hitgroup_prog_group, d_hit_data.get());
    }

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

            s_pathtracer->impl->output_texture->setData(
                atcg::createDeviceTensorFromPointer((uint8_t*)data_ptr,
                                                    {s_pathtracer->impl->height, s_pathtracer->impl->width, 4}));
            s_pathtracer->impl->dirty = false;
        }
    }

    return s_pathtracer->impl->output_texture;
}

}    // namespace atcg