#include <Pathtracing/Shader/PathtracingShader.h>

#include <DataStructure/TorchUtils.h>

#include <Scene/Components.h>
#include <Scene/Entity.h>
#include <Pathtracing/Tracing.h>

#include <thread>
#include <mutex>
#include <execution>

namespace atcg
{
PathtracingShader::PathtracingShader() : CPURaytracingShader()
{
    setTensor("HDR", torch::zeros({1, 1, 3}, atcg::TensorOptions::floatDeviceOptions()));
}

PathtracingShader::~PathtracingShader()
{
    reset();
}

void PathtracingShader::initializePipeline()
{
    _diffuse_images.clear();
    _roughness_images.clear();
    _metallic_images.clear();
    _positions.clear();
    _normals.clear();
    _uvs.clear();
    _faces.clear();
    _mesh_idx.clear();

    auto view = _scene->getAllEntitiesWith<GeometryComponent, MeshRenderComponent, TransformComponent>();

    uint32_t total_vertices = 0;
    uint32_t total_faces    = 0;
    std::vector<atcg::ref_ptr<Graph>> geometry;
    std::vector<TransformComponent> transforms;
    for(auto e: view)
    {
        Entity entity(e, _scene.get());

        TransformComponent transform = entity.getComponent<TransformComponent>();

        transforms.push_back(transform);

        atcg::ref_ptr<Graph> graph = entity.getComponent<GeometryComponent>().graph;
        geometry.push_back(graph);

        total_vertices += graph->n_vertices();
        total_faces += graph->n_faces();

        auto& material = entity.getComponent<MeshRenderComponent>().material;

        _diffuse_images.push_back(atcg::make_ref<Image>(material.getDiffuseTexture()));
        _roughness_images.push_back(atcg::make_ref<Image>(material.getRoughnessTexture()));
        _metallic_images.push_back(atcg::make_ref<Image>(material.getMetallicTexture()));
    }

    _positions.resize(total_vertices);
    _normals.resize(total_vertices);
    _uvs.resize(total_vertices);
    _faces.resize(total_faces);
    _mesh_idx.resize(total_faces);

    uint32_t vertex_idx = 0;
    uint32_t face_idx   = 0;

    for(int i = 0; i < geometry.size(); ++i)
    {
        auto& graph      = geometry[i];
        Vertex* vertices = graph->getVerticesBuffer()->getHostPointer<Vertex>();

        TransformComponent& transform = transforms[i];
        glm::mat4 model               = transform.getModel();
        glm::mat4 normal_matrix       = glm::transpose(glm::inverse(model));

        glm::u32vec3* faces = graph->getFaceIndexBuffer()->getHostPointer<glm::u32vec3>();

        for(int j = 0; j < graph->n_faces(); ++j)
        {
            _faces[face_idx]    = faces[j] + vertex_idx;
            _mesh_idx[face_idx] = i;
            ++face_idx;
        }

        for(int j = 0; j < graph->n_vertices(); ++j)
        {
            _positions[vertex_idx] = model * glm::vec4(vertices[j].position, 1.0f);
            _normals[vertex_idx]   = glm::normalize(glm::vec3(normal_matrix * glm::vec4(vertices[j].normal, 0.0f)));
            _uvs[vertex_idx]       = vertices[j].uv;
            ++vertex_idx;
        }

        graph->getVerticesBuffer()->unmapHostPointers();
        graph->getFaceIndexBuffer()->unmapHostPointers();
    }

    _accel = atcg::make_ref<BVHAccelerationStructure>();

    nanort::BVHAccel<float> accel;
    nanort::TriangleMesh<float> triangle_mesh(reinterpret_cast<const float*>(_positions.data()),
                                              reinterpret_cast<const uint32_t*>(_faces.data()),
                                              sizeof(float) * 3);
    nanort::TriangleSAHPred<float> triangle_pred(reinterpret_cast<const float*>(_positions.data()),
                                                 reinterpret_cast<const uint32_t*>(_faces.data()),
                                                 sizeof(float) * 3);
    bool ret = accel.Build(total_faces, triangle_mesh, triangle_pred);
    assert(ret);

    nanort::BVHBuildStatistics stats = accel.GetStatistics();

    ATCG_INFO("BVH statistics:");
    ATCG_INFO("\t# of leaf   nodes: {0}", stats.num_leaf_nodes);
    ATCG_INFO("\t# of branch nodes: {0}", stats.num_branch_nodes);
    ATCG_INFO("\tMax tree depth   : {0}", stats.max_tree_depth);

    _accel->setBVH(accel);

    _hasSkybox = Renderer::hasSkybox();
    if(_hasSkybox)
    {
        atcg::ref_ptr<Texture2D> skybox_texture = atcg::Renderer::getSkyboxTexture();

        _skybox_image = atcg::make_ref<Image>(skybox_texture);
    }
}

void PathtracingShader::reset()
{
    _frame_counter = 0;
}

void PathtracingShader::setCamera(const atcg::ref_ptr<PerspectiveCamera>& camera)
{
    _camera          = camera;
    _inv_camera_view = glm::inverse(camera->getView());
    _fov_y           = camera->getFOV();
}

glm::vec3 PathtracingShader::read_image(const atcg::ref_ptr<Image>& image, const glm::vec2& uv)
{
    glm::vec4 result(0);

    glm::ivec2 pixel(uv.x * image->width(), image->height() - uv.y * image->height());
    pixel = glm::clamp(pixel, glm::ivec2(0), glm::ivec2(image->width() - 1, image->height() - 1));

    glm::vec3 color;

    uint32_t channels = image->channels();
    if(image->isHDR())
    {
        if(channels == 4)
        {
            color = image->dataAt<glm::vec4>(pixel.x + image->width() * pixel.y);
        }
        else if(channels == 3)
        {
            color = image->dataAt<glm::vec3>(pixel.x + image->width() * pixel.y);
        }
        else
        {
            color = glm::vec3(image->dataAt<float>(pixel.x + image->width() * pixel.y));
        }
    }
    else
    {
        if(channels == 4)
        {
            glm::u8vec4 val = image->dataAt<glm::u8vec4>(pixel.x + image->width() * pixel.y);
            color           = glm::vec3((float)val.x, (float)val.y, (float)val.z) / 255.0f;
        }
        else if(channels == 3)
        {
            glm::u8vec3 val = image->dataAt<glm::u8vec3>(pixel.x + image->width() * pixel.y);
            color           = glm::vec3((float)val.x, (float)val.y, (float)val.z) / 255.0f;
        }
        else
        {
            uint8_t val = image->dataAt<uint8_t>(pixel.x + image->width() * pixel.y);
            color       = glm::vec3((float)val) / 255.0f;
        }
    }

    return color;
}

void PathtracingShader::generateRays(torch::Tensor& output)
{
    auto accumulation_buffer = getTensor("HDR");
    if(accumulation_buffer.ndimension() != 3 || accumulation_buffer.size(0) != output.size(0) ||
       accumulation_buffer.size(1) != output.size(1))
    {
        accumulation_buffer =
            torch::zeros({output.size(0), output.size(1), 3}, atcg::TensorOptions::floatDeviceOptions());
        setTensor("HDR", accumulation_buffer);

        _horizontalScanLine.resize(output.size(1));
        _verticalScanLine.resize(output.size(0));
        for(int i = 0; i < output.size(1); ++i)
        {
            _horizontalScanLine[i] = i;
        }

        for(int i = 0; i < output.size(0); ++i)
        {
            _verticalScanLine[i] = i;
        }
    }

    glm::mat4 camera_view = _inv_camera_view;
    glm::vec3 cam_eye_    = camera_view[3];
    glm::vec3 U_          = glm::normalize(camera_view[0]);
    glm::vec3 V_          = glm::normalize(camera_view[1]);
    glm::vec3 W_          = -glm::normalize(camera_view[2]);

    int max_trace_depth = 10;

    int width  = output.size(1);
    int height = output.size(0);

    std::for_each(
        std::execution::par,
        _verticalScanLine.begin(),
        _verticalScanLine.end(),
        [&](uint32_t y)
        {
            std::for_each(
                std::execution::par,
                _horizontalScanLine.begin(),
                _horizontalScanLine.end(),
                [&](uint32_t x)
                {
                    uint64_t seed = sampleTEA64(x + width * y, _frame_counter);
                    atcg::PCG32 rng(seed);

                    glm::vec2 jitter = rng.next2d();
                    float u          = (((float)x + jitter.x) / (float)width - 0.5f) * 2.0f;
                    float v          = (((float)(height - y) + jitter.y) / (float)height - 0.5f) * 2.0f;

                    glm::vec3 U = U_ * (float)width / (float)height;
                    glm::vec3 V = V_;
                    glm::vec3 W = W_ / glm::tan(glm::radians(_fov_y / 2.0f));

                    glm::vec3 ray_dir    = glm::normalize(u * U + v * V + W);
                    glm::vec3 ray_origin = cam_eye_;
                    glm::vec3 radiance(0);
                    glm::vec3 throughput(1);

                    glm::vec3 next_origin;
                    glm::vec3 next_dir;

                    for(int n = 0; n < max_trace_depth; ++n)
                    {
#define CREATE_TENSOR(vector, type) atcg::createHostTensorFromPointer((type*)vector.data(), {(int)vector.size(), 3})
                        SurfaceInteraction si = Tracing::traceRay(_accel,
                                                                  CREATE_TENSOR(_positions, float),
                                                                  CREATE_TENSOR(_normals, float),
                                                                  CREATE_TENSOR(_uvs, float),
                                                                  CREATE_TENSOR(_faces, int32_t),
                                                                  ray_origin,
                                                                  ray_dir,
                                                                  1e-3f,
                                                                  1e6f);
#undef CREATE_TENSOR
                        if(si.valid)
                        {
                            // PBR Sampling
                            uint32_t mesh_index     = _mesh_idx[si.primitive_idx];
                            glm::vec3 diffuse_color = read_image(_diffuse_images[mesh_index], si.uv);
                            float metallic          = read_image(_metallic_images[mesh_index], si.uv).x;
                            float roughness         = read_image(_roughness_images[mesh_index], si.uv).x;

                            glm::vec3 metallic_color = (1.0f - metallic) * glm::vec3(0.04) + metallic * diffuse_color;

                            BSDFSamplingResult result =
                                sampleGGX(si, diffuse_color, metallic_color, metallic, roughness, rng);
                            if(result.sample_probability > 0.0f)
                            {
                                next_origin = si.position;
                                next_dir    = result.out_dir;
                                throughput *= result.bsdf_weight;
                            }
                            else
                            {
                                break;
                            }
                        }
                        else
                        {
                            float theta = std::acos(ray_dir.y) / glm::pi<float>();
                            float phi =
                                (std::atan2(ray_dir.z, ray_dir.x) + glm::pi<float>()) / (2.0f * glm::pi<float>());

                            if(_hasSkybox) radiance = throughput * read_image(_skybox_image, glm::vec2(phi, theta));

                            break;
                        }

                        ray_origin = next_origin;
                        ray_dir    = next_dir;
                    }

                    if(_frame_counter > 0)
                    {
                        // Mix with previous subframes if present!
                        const float a = 1.0f / static_cast<float>(_frame_counter + 1);
                        const glm::vec3 prev_output_radiance =
                            ((const glm::vec3*)accumulation_buffer.data_ptr())[x + width * y];
                        radiance = glm::lerp(prev_output_radiance, radiance, a);
                    }

                    ((glm::vec3*)accumulation_buffer.data_ptr())[x + width * y] = radiance;

                    glm::vec3 tone_mapped = glm::clamp(glm::pow(1.0f - glm::exp(-radiance), glm::vec3(1.0f / 2.2f)),
                                                       glm::vec3(0),
                                                       glm::vec3(1));

                    ((glm::u8vec4*)output.data_ptr())[x + width * y] = glm::u8vec4((uint8_t)(tone_mapped.x * 255.0f),
                                                                                   (uint8_t)(tone_mapped.y * 255.0f),
                                                                                   (uint8_t)(tone_mapped.z * 255.0f),
                                                                                   255);
                });
        });
    ++_frame_counter;
}
}    // namespace atcg