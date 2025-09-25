#include <Renderer/Texture.h>
#include <DataStructure/Image.h>
#include <Core/CUDA.h>
#include <Core/glm.h>
#include <DataStructure/CUDATexture.h>

__global__ void fillSurf(atcg::CUDATexture<glm::u8vec3> tex)
{
    size_t tid = atcg::threadIndex();
    if(tid >= tex.spec.width * tex.spec.height) return;

    uint32_t x = tid % tex.spec.width;
    uint32_t y = tid / tex.spec.width;

    uint8_t u = uint8_t((float)x / (float)tex.spec.width * 255.0f);
    uint8_t v = uint8_t((float)y / (float)tex.spec.height * 255.0f);

    tex.write(glm::u8vec3(u, v, 0), glm::ivec2(x, y));
}

__global__ void read(atcg::CUDATexture<glm::u8vec3> tex)
{
    size_t tid = atcg::threadIndex();
    if(tid >= tex.spec.width * tex.spec.height) return;

    uint32_t x = tid % tex.spec.width;
    uint32_t y = tid / tex.spec.width;

    if(x == 512 && y == 512)
    {
        float u       = (float)x / (float)tex.spec.width;
        float v       = (float)y / (float)tex.spec.height;
        glm::vec4 val = tex.read(glm::vec2(u, v));

        printf("val: %f %f %f\n", val.x, val.y, val.z);
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

    atcg::CUDATexture<glm::u8vec3> tex = {};
    tex.spec                           = spec;
    // tex.texture_data.raw.data          = texture_data.data_ptr();
    tex.texture_data.texture = texture->getTextureObject();
    tex.texture_data.surface = texture->getSurfaceObject();
    tex.default_value        = glm::vec4(1);

    fillSurf<<<blocks, 128>>>(tex);
    SYNCHRONIZE_DEFAULT_STREAM();

    read<<<blocks, 128>>>(tex);
    SYNCHRONIZE_DEFAULT_STREAM();


    // texture->setData(texture_data);
    texture->unmapDevicePointers();

    atcg::Image img(texture->getData(atcg::CPU));
    img.store("Test.png");
}