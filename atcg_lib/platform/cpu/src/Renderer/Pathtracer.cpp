#include <Renderer/Pathtracer.h>

#include <Scene/Components.h>
#include <Math/Tracing.h>
#include <DataStructure/Timer.h>
#include <thread>
#include <mutex>
#include <Math/Random.h>
#include <execution>

namespace atcg
{

Pathtracer* Pathtracer::s_pathtracer = new Pathtracer;

struct BSDFSamplingResult
{
    glm::vec3 out_dir;
    glm::vec3 bsdf_weight;
    float sample_probability = 0.0f;
};

struct SurfaceInteraction
{
    bool valid = false;
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 barys;
    glm::vec2 uv;
    glm::vec3 incoming_direction;
    float incoming_distance;
    uint32_t primitive_idx;
};

class Pathtracer::Impl
{
public:
    Impl() {}
    ~Impl() {}

    void worker();

    // Baked scene

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> uvs;
    std::vector<glm::u32vec3> faces;
    std::vector<uint32_t> mesh_idx;
    nanort::BVHAccel<float> accel;

    std::vector<atcg::ref_ptr<Image>> diffuse_images;
    std::vector<atcg::ref_ptr<Image>> roughness_images;
    std::vector<atcg::ref_ptr<Image>> metallic_images;

    atcg::ref_ptr<PerspectiveCamera> camera;
    bool hasSkybox = false;
    atcg::ref_ptr<Image> skybox_image;
    glm::vec3 read_image(const atcg::ref_ptr<Image>& image, const glm::vec2& uv);
    BSDFSamplingResult sampleGGX(const SurfaceInteraction& si,
                                 const glm::vec3& diffuse_color,
                                 const glm::vec3& specular_F0,
                                 const float& metallic,
                                 const float& roughness,
                                 PCG32& rng);

    SurfaceInteraction traceRay(const glm::vec3& origin, const glm::vec3& dir, float tmin, float tmax);

    std::vector<glm::vec3> accumulation_buffer;

    // Basic swap chain
    glm::u8vec4* output_buffer;
    uint8_t swap_index = 1;
    atcg::ref_ptr<glm::u8vec4> swap_chain_buffer;
    bool dirty = false;
    std::mutex swap_chain_mutex;
    std::vector<uint32_t> horizontalScanLine;
    std::vector<uint32_t> verticalScanLine;

    atcg::ref_ptr<atcg::Texture2D> output_texture;

    std::thread worker_thread;

    std::atomic_bool running = false;
};

glm::vec3 Pathtracer::Impl::read_image(const atcg::ref_ptr<Image>& image, const glm::vec2& uv)
{
    glm::vec4 result(0);

    glm::ivec2 pixel(uv.x * image->width(), image->height() - uv.y * image->height());
    pixel = glm::clamp(pixel, glm::ivec2(0), glm::ivec2(image->width() - 1, image->height() - 1));

    glm::vec3 color;

    uint32_t channels = image->channels();
    if(image->isHDR())
    {
        if(channels == 4) { color = image->dataAt<glm::vec4>(pixel.x + image->width() * pixel.y); }
        else { color = glm::vec3(image->dataAt<float>(pixel.x + image->width() * pixel.y)); }
    }
    else
    {
        if(channels == 4)
        {
            glm::u8vec4 val = image->dataAt<glm::u8vec4>(pixel.x + image->width() * pixel.y);
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

inline glm::mat3 compute_local_frame(glm::vec3 localZ)
{
    float x  = localZ.x;
    float y  = localZ.y;
    float z  = localZ.z;
    float sz = (z >= 0) ? 1 : -1;
    float a  = 1 / (sz + z);
    float ya = y * a;
    float b  = x * ya;
    float c  = x * sz;

    glm::vec3 localX = glm::vec3(c * x * a - 1, sz * b, c);
    glm::vec3 localY = glm::vec3(b, y * ya - sz, y);

    glm::mat3 frame;
    // Set columns of matrix
    frame[0] = localX;
    frame[1] = localY;
    frame[2] = localZ;
    return frame;
}

inline glm::vec3 warp_square_to_hemisphere_ggx(const glm::vec2& uv, float roughness)
{
    // GGX NDF sampling
    float cos_theta = glm::sqrt((1.0f - uv.x) / (1.0f + (roughness * roughness - 1.0f) * uv.x));
    float sin_theta = glm::sqrt(glm::max(0.0f, 1.0f - cos_theta * cos_theta));
    float phi       = 2.0f * glm::pi<float>() * uv.y;

    float x = sin_theta * glm::cos(phi);
    float y = sin_theta * glm::sin(phi);
    float z = cos_theta;

    return glm::vec3(x, y, z);
}

inline glm::vec3 warp_square_to_hemisphere_cosine(const glm::vec2& uv)
{
    // Sample disk uniformly
    float r   = glm::sqrt(uv.x);
    float phi = 2.0f * glm::pi<float>() * uv.y;

    // Project disk sample onto hemisphere
    float x = r * glm::cos(phi);
    float y = r * glm::sin(phi);
    float z = glm::sqrt(glm::max(0.0f, 1 - uv.x));

    return glm::vec3(x, y, z);
}

inline float warp_normal_to_reflected_direction_pdf(const glm::vec3& reflected_dir, const glm::vec3& normal)
{
    return 1 / glm::abs(4 * glm::dot(reflected_dir, normal));
}

inline float fresnel_schlick(const float F0, const float VdotH)
{
    return F0 + (1.0f - F0) * glm::pow(glm::max(0.0f, 1.0f - VdotH), 5.0f);
}

inline glm::vec3 fresnel_schlick(const glm::vec3 F0, const float VdotH)
{
    return F0 + (glm::vec3(1.0f) - F0) * glm::pow(glm::max(0.0f, 1.0f - VdotH), 5.0f);
}


inline float D_GGX(const float NdotH, const float roughness)
{
    float a2 = roughness * roughness;
    float d  = (NdotH * a2 - NdotH) * NdotH + 1.0f;
    return a2 / (glm::pi<float>() * d * d);
}

float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float geometrySmith(float NdotL, float NdotV, float roughness)
{
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

BSDFSamplingResult Pathtracer::Impl::sampleGGX(const SurfaceInteraction& si,
                                               const glm::vec3& diffuse_color,
                                               const glm::vec3& specular_F0,
                                               const float& metallic,
                                               const float& roughness,
                                               PCG32& rng)
{
    BSDFSamplingResult result;

    // Direction towards viewer
    glm::vec3 view_dir = -si.incoming_direction;
    glm::vec3 normal   = si.normal;

    // Don't trace a new ray if surface is viewed from below
    float NdotV = glm::dot(normal, view_dir);
    if(NdotV <= 0) { return result; }

    // The matrix local_frame transforms a vector from the coordinate system where geom.N corresponds to the z-axis to
    // the world coordinate system.
    glm::mat3 local_frame = compute_local_frame(normal);

    float diffuse_probability = glm::dot(diffuse_color, glm::vec3(1)) /
                                (glm::dot(diffuse_color, glm::vec3(1)) + glm::dot(specular_F0, glm::vec3(1)) + 1e-5f);
    float specular_probability = 1 - diffuse_probability;

    if(rng.next1d() < diffuse_probability)
    {
        // Sample light direction from diffuse bsdf
        glm::vec3 local_outgoing_ray_dir = warp_square_to_hemisphere_cosine(rng.next2d());
        // Transform local outgoing direction from tangent space to world space
        result.out_dir = local_frame * local_outgoing_ray_dir;
    }
    else
    {
        // Sample light direction from specular bsdf
        glm::vec3 local_halfway = warp_square_to_hemisphere_ggx(rng.next2d(), roughness);
        // Transform local halfway vector from tangent space to world space
        glm::vec3 halfway = local_frame * local_halfway;
        result.out_dir    = glm::reflect(si.incoming_direction, halfway);
    }

    // It is possible that light directions below the horizon are sampled..
    // If outgoing ray direction is below horizon, let the sampling fail!
    float NdotL = glm::dot(normal, result.out_dir);
    if(NdotL <= 0)
    {
        result.sample_probability = 0;
        return result;
    }

    glm::vec3 diffuse_bsdf = diffuse_color / glm::pi<float>();
    float diffuse_pdf      = NdotL / glm::pi<float>();

    glm::vec3 specular_bsdf = glm::vec3(0);
    float specular_pdf      = 0;
    // Only compute specular component if specular_f0 is not zero!
    glm::vec3 kD(1.0f);
    if(glm::dot(specular_F0, specular_F0) > 1e-6)
    {
        glm::vec3 halfway = glm::normalize(result.out_dir + view_dir);
        float HdotV       = glm::dot(halfway, result.out_dir);
        float NdotH       = glm::dot(halfway, normal);

        // Normal distribution
        float NDF = D_GGX(NdotH, roughness);

        // Visibility
        float G = geometrySmith(NdotL, NdotV, roughness);

        // Fresnel
        glm::vec3 F = fresnel_schlick(specular_F0, HdotV);

        kD = (1.0f - F) * (1.0f - metallic);

        glm::vec3 numerator = NDF * G * F;
        float denominator   = 4.0f * NdotV * NdotL + 1e-5f;
        specular_bsdf       = numerator / denominator;

        float halfway_pdf = NDF * NdotH;
        float halfway_to_outgoing_pdf =
            warp_normal_to_reflected_direction_pdf(result.out_dir, halfway);    // 1 / (4*HdotV)
        specular_pdf = halfway_pdf * halfway_to_outgoing_pdf;
    }

    result.sample_probability = diffuse_probability * diffuse_pdf + specular_probability * specular_pdf + 1e-5f;
    result.bsdf_weight        = (specular_bsdf + kD * diffuse_bsdf) * NdotL / result.sample_probability;

    return result;
}

SurfaceInteraction Pathtracer::Impl::traceRay(const glm::vec3& origin, const glm::vec3& dir, float tmin, float tmax)
{
    SurfaceInteraction si;
    si.incoming_direction = dir;

    nanort::Ray<float> ray;
    memcpy(ray.org, glm::value_ptr(origin), sizeof(glm::vec3));
    memcpy(ray.dir, glm::value_ptr(dir), sizeof(glm::vec3));

    ray.min_t = tmin;
    ray.max_t = tmax;

    nanort::TriangleIntersector<> triangle_intersector(reinterpret_cast<const float*>(positions.data()),
                                                       reinterpret_cast<const uint32_t*>(faces.data()),
                                                       sizeof(float) * 3);
    nanort::TriangleIntersection<> isect;
    bool hit = accel.Traverse(ray, triangle_intersector, &isect);

    if(!hit) { return si; }

    si.valid         = true;
    si.position      = origin + isect.t * dir;
    si.barys         = glm::vec2(isect.u, isect.v);
    si.primitive_idx = isect.prim_id;

    glm::u32vec3 face = faces[isect.prim_id];

    si.normal = (1.0f - isect.u - isect.v) * normals[face.x] + isect.u * normals[face.y] + isect.v * normals[face.z];

    si.uv                = (1.0f - isect.u - isect.v) * uvs[face.x] + isect.u * uvs[face.y] + isect.v * uvs[face.z];
    si.incoming_distance = isect.t;

    return si;
}

void Pathtracer::Impl::worker()
{
    uint32_t frame_counter = 0;
    while(true)
    {
        Timer frame_time;

        glm::mat4 camera_view = glm::inverse(camera->getView());
        glm::vec3 cam_eye     = camera_view[3];
        glm::vec3 U           = glm::normalize(camera_view[0]);
        glm::vec3 V           = glm::normalize(camera_view[1]);
        glm::vec3 W           = -glm::normalize(camera_view[2]);

        int max_trace_depth = 10;

        std::for_each(
            std::execution::par,
            verticalScanLine.begin(),
            verticalScanLine.end(),
            [&](uint32_t y)
            {
                std::for_each(
                    std::execution::par,
                    horizontalScanLine.begin(),
                    horizontalScanLine.end(),
                    [&](uint32_t x)
                    {
                        uint64_t seed = sampleTEA64(x + 512 * y, frame_counter);
                        atcg::PCG32 rng(seed);

                        glm::vec2 jitter = rng.next2d();
                        float u          = (((float)x + jitter.x) / 512.0f - 0.5f) * 2.0f;
                        float v          = (((float)(512 - y) + jitter.y) / 512.0f - 0.5f) * 2.0f;

                        glm::vec3 ray_dir    = glm::normalize(u * U + v * V + W);
                        glm::vec3 ray_origin = cam_eye;
                        glm::vec3 radiance(0);
                        glm::vec3 throughput(1);

                        glm::vec3 next_origin;
                        glm::vec3 next_dir;

                        for(int n = 0; n < max_trace_depth; ++n)
                        {
                            SurfaceInteraction si = s_pathtracer->impl->traceRay(ray_origin, ray_dir, 1e-3f, 1e6f);
                            if(si.valid)
                            {
                                // PBR Sampling
                                uint32_t mesh_index     = mesh_idx[si.primitive_idx];
                                glm::vec3 diffuse_color = read_image(diffuse_images[mesh_index], si.uv);
                                float metallic          = read_image(metallic_images[mesh_index], si.uv).x;
                                float roughness         = read_image(roughness_images[mesh_index], si.uv).x;

                                glm::vec3 metallic_color =
                                    (1.0f - metallic) * glm::vec3(0.04) + metallic * diffuse_color;

                                BSDFSamplingResult result =
                                    sampleGGX(si, diffuse_color, metallic_color, metallic, roughness, rng);
                                if(result.sample_probability > 0.0f)
                                {
                                    next_origin = si.position;
                                    next_dir    = result.out_dir;
                                    throughput *= result.bsdf_weight;
                                }
                                else { break; }
                            }
                            else
                            {
                                float theta = std::acos(ray_dir.y) / glm::pi<float>();
                                float phi =
                                    (std::atan2(ray_dir.z, ray_dir.x) + glm::pi<float>()) / (2.0f * glm::pi<float>());

                                if(hasSkybox) radiance = throughput * read_image(skybox_image, glm::vec2(phi, theta));

                                break;
                            }

                            ray_origin = next_origin;
                            ray_dir    = next_dir;
                        }

                        if(frame_counter > 0)
                        {
                            // Mix with previous subframes if present!
                            const float a                        = 1.0f / static_cast<float>(frame_counter + 1);
                            const glm::vec3 prev_output_radiance = accumulation_buffer[x + 512 * y];
                            radiance                             = glm::lerp(prev_output_radiance, radiance, a);
                        }

                        accumulation_buffer[x + 512 * y] = radiance;

                        glm::vec3 tone_mapped = glm::clamp(glm::pow(1.0f - glm::exp(-radiance), glm::vec3(1.0f / 2.2f)),
                                                           glm::vec3(0),
                                                           glm::vec3(1));

                        output_buffer[x + 512 * y] = glm::vec4((uint8_t)(tone_mapped.x * 255.0f),
                                                               (uint8_t)(tone_mapped.y * 255.0f),
                                                               (uint8_t)(tone_mapped.z * 255.0f),
                                                               255);
                    });
            });

        ++frame_counter;
        std::cout << frame_time.elapsedMillis() << "\n";

        // Perform swap
        {
            std::lock_guard guard(swap_chain_mutex);
            output_buffer = swap_chain_buffer.get() + swap_index * 512 * 512;
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

    // TODO: Hard coded 512 x 512 output image for now
    s_pathtracer->impl->swap_chain_buffer = atcg::ref_ptr<glm::u8vec4>(2 * 512 * 512);    // 2 swap chain buffers
    s_pathtracer->impl->output_buffer     = s_pathtracer->impl->swap_chain_buffer.get();

    s_pathtracer->impl->accumulation_buffer.resize(512 * 512);
    memset(s_pathtracer->impl->accumulation_buffer.data(),
           0,
           sizeof(glm::vec3) * s_pathtracer->impl->accumulation_buffer.size());

    TextureSpecification spec;
    spec.width                         = 512;
    spec.height                        = 512;
    s_pathtracer->impl->output_texture = atcg::Texture2D::create(spec);

    s_pathtracer->impl->horizontalScanLine.resize(512);
    s_pathtracer->impl->verticalScanLine.resize(512);
    for(int i = 0; i < 512; ++i)
    {
        s_pathtracer->impl->horizontalScanLine[i] = i;
        s_pathtracer->impl->verticalScanLine[i]   = i;
    }
}

void Pathtracer::bakeScene(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<PerspectiveCamera>& camera)
{
    s_pathtracer->impl->diffuse_images.clear();
    s_pathtracer->impl->roughness_images.clear();
    s_pathtracer->impl->metallic_images.clear();
    s_pathtracer->impl->positions.clear();
    s_pathtracer->impl->normals.clear();
    s_pathtracer->impl->uvs.clear();
    s_pathtracer->impl->faces.clear();
    s_pathtracer->impl->mesh_idx.clear();

    s_pathtracer->impl->camera = camera;

    auto view = scene->getAllEntitiesWith<GeometryComponent, MeshRenderComponent, TransformComponent>();

    uint32_t total_vertices = 0;
    uint32_t total_faces    = 0;
    std::vector<atcg::ref_ptr<Graph>> geometry;
    std::vector<TransformComponent> transforms;
    for(auto e: view)
    {
        Entity entity(e, scene.get());

        TransformComponent transform = entity.getComponent<TransformComponent>();

        transforms.push_back(transform);

        atcg::ref_ptr<Graph> graph = entity.getComponent<GeometryComponent>().graph;
        geometry.push_back(graph);

        total_vertices += graph->n_vertices();
        total_faces += graph->n_faces();

        auto& material = entity.getComponent<MeshRenderComponent>().material;

        s_pathtracer->impl->diffuse_images.push_back(atcg::make_ref<Image>(material.getDiffuseTexture()));
        s_pathtracer->impl->roughness_images.push_back(atcg::make_ref<Image>(material.getRoughnessTexture()));
        s_pathtracer->impl->metallic_images.push_back(atcg::make_ref<Image>(material.getMetallicTexture()));
    }

    s_pathtracer->impl->positions.resize(total_vertices);
    s_pathtracer->impl->normals.resize(total_vertices);
    s_pathtracer->impl->uvs.resize(total_vertices);
    s_pathtracer->impl->faces.resize(total_faces);
    s_pathtracer->impl->mesh_idx.resize(total_faces);

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
            s_pathtracer->impl->faces[face_idx]    = faces[j] + vertex_idx;
            s_pathtracer->impl->mesh_idx[face_idx] = i;
            ++face_idx;
        }

        for(int j = 0; j < graph->n_vertices(); ++j)
        {
            s_pathtracer->impl->positions[vertex_idx] = model * glm::vec4(vertices[j].position, 1.0f);
            s_pathtracer->impl->normals[vertex_idx] =
                glm::normalize(glm::vec3(normal_matrix * glm::vec4(vertices[j].normal, 0.0f)));
            s_pathtracer->impl->uvs[vertex_idx] = vertices[j].uv;
            ++vertex_idx;
        }

        graph->getVerticesBuffer()->unmapHostPointers();
        graph->getFaceIndexBuffer()->unmapHostPointers();
    }

    nanort::TriangleMesh<float> triangle_mesh(reinterpret_cast<const float*>(s_pathtracer->impl->positions.data()),
                                              reinterpret_cast<const uint32_t*>(s_pathtracer->impl->faces.data()),
                                              sizeof(float) * 3);
    nanort::TriangleSAHPred<float> triangle_pred(reinterpret_cast<const float*>(s_pathtracer->impl->positions.data()),
                                                 reinterpret_cast<const uint32_t*>(s_pathtracer->impl->faces.data()),
                                                 sizeof(float) * 3);
    bool ret = s_pathtracer->impl->accel.Build(total_faces, triangle_mesh, triangle_pred);
    assert(ret);

    nanort::BVHBuildStatistics stats = s_pathtracer->impl->accel.GetStatistics();

    ATCG_INFO("BVH statistics:");
    ATCG_INFO("\t# of leaf   nodes: {0}", stats.num_leaf_nodes);
    ATCG_INFO("\t# of branch nodes: {0}", stats.num_branch_nodes);
    ATCG_INFO("\tMax tree depth   : {0}", stats.max_tree_depth);

    s_pathtracer->impl->hasSkybox = Renderer::hasSkybox();
    if(s_pathtracer->impl->hasSkybox)
    {
        atcg::ref_ptr<Texture2D> skybox_texture = atcg::Renderer::getSkyboxTexture();

        s_pathtracer->impl->skybox_image = atcg::make_ref<Image>(skybox_texture);
    }
}

void Pathtracer::start()
{
    if(!s_pathtracer->impl->running)
    {
        memset(s_pathtracer->impl->accumulation_buffer.data(),
               0,
               sizeof(glm::vec3) * s_pathtracer->impl->accumulation_buffer.size());
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
        std::lock_guard(s_pathtracer->impl->swap_chain_mutex);

        if(s_pathtracer->impl->dirty)
        {
            glm::u8vec4* data_ptr =
                s_pathtracer->impl->swap_chain_buffer.get() + s_pathtracer->impl->swap_index * 512 * 512;
            s_pathtracer->impl->output_texture->setData((void*)data_ptr);
            s_pathtracer->impl->dirty = false;
        }
    }

    return s_pathtracer->impl->output_texture;
}

}    // namespace atcg