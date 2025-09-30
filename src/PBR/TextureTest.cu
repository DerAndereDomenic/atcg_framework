#include <Renderer/Texture.h>
#include <DataStructure/Image.h>
#include <Core/CUDA.h>
#include <Core/glm.h>
#include <DataStructure/CUDATexture.h>
#include <DataStructure/TextureSampler.h>

__global__ void fillSurf(atcg::TextureSampler<glm::vec2> tex)
{
    size_t tid = atcg::threadIndex();
    if(tid >= tex.getSpecification().width * tex.getSpecification().height) return;

    uint32_t x = tid % tex.getSpecification().width;
    uint32_t y = tid / tex.getSpecification().width;

    float u = (float)x / (float)tex.getSpecification().width;
    float v = (float)y / (float)tex.getSpecification().height;

    tex.write(glm::vec2(u, v), glm::ivec2(x, y));
}

__global__ void read(atcg::TextureSampler<glm::vec2> tex)
{
    size_t tid = atcg::threadIndex();
    if(tid >= tex.getSpecification().width * tex.getSpecification().height) return;

    uint32_t x = tid % tex.getSpecification().width;
    uint32_t y = tid / tex.getSpecification().width;

    if(x == 512 && y == 512)
    {
        float u       = (float)x / (float)tex.getSpecification().width;
        float v       = (float)y / (float)tex.getSpecification().height;
        glm::vec2 val = tex.read(glm::vec2(u, v));

        printf("val: %f %f\n", val.x, val.y);
    }
}

void test()
{
    atcg::TextureSpecification spec;

    spec.width  = 1024;
    spec.height = 1024;
    spec.format = atcg::TextureFormat::RGB;

    auto texture = atcg::Texture2D::create(spec);

    auto blocks = atcg::configure(spec.width * spec.height, 128);

    auto data = texture->getData(atcg::GPU);

    atcg::TextureSampler<glm::vec2> tex(data.data_ptr(), spec);

    // atcg::CUDATexture<glm::u8vec3> tex = {};
    // tex.spec                           = spec;
    // // tex.texture_data.raw.data          = texture_data.data_ptr();
    // tex.texture_data.texture = texture->getTextureObject();
    // tex.texture_data.surface = texture->getSurfaceObject();
    // tex.default_value        = glm::vec4(1);

    fillSurf<<<blocks, 128>>>(tex);
    SYNCHRONIZE_DEFAULT_STREAM();

    read<<<blocks, 128>>>(tex);
    SYNCHRONIZE_DEFAULT_STREAM();

    texture->setData(data);
    // texture->setData(texture_data);
    texture->unmapDevicePointers();

    atcg::Image img(texture->getData(atcg::CPU));
    img.store("Test.png");
}