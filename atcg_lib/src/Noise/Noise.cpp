#include <Noise/Noise.h>

#include <Renderer/ShaderManager.h>

namespace atcg
{
namespace Noise
{

atcg::ref_ptr<Texture2D> createWhiteNoiseTexture2D(glm::ivec2 dim)
{
    atcg::ref_ptr<Shader> compute_shader = ShaderManager::getShader("white_noise_2D");

    atcg::ref_ptr<Texture2D> result = Texture2D::createFloatTexture(dim.x, dim.y);

    result->useForCompute();

    // Use 8x8x1 = 64 thread sized work group
    compute_shader->dispatch(glm::ivec3(ceil(dim.x / 8), ceil(dim.y / 8), 1));

    return result;
}

atcg::ref_ptr<Texture3D> createWhiteNoiseTexture3D(glm::ivec3 dim)
{
    atcg::ref_ptr<Shader> compute_shader = ShaderManager::getShader("white_noise_3D");

    atcg::ref_ptr<Texture3D> result = Texture3D::createFloatTexture(dim.x, dim.y, dim.z);

    result->useForCompute();

    // Use 4x4x4 = 64 thread sized work group
    compute_shader->dispatch(glm::ivec3(ceil(dim.x / 4), ceil(dim.y / 4), ceil(dim.z / 4)));

    return result;
}

atcg::ref_ptr<Texture2D> createWorleyNoiseTexture2D(glm::ivec2 dim, uint32_t num_points)
{
    atcg::ref_ptr<Shader> compute_shader = ShaderManager::getShader("worly_noise_2D");

    atcg::ref_ptr<Texture2D> result = Texture2D::createFloatTexture(dim.x, dim.y);


    // Use 8x8x1 = 64 thread sized work group
    compute_shader->use();
    compute_shader->setInt("num_points", num_points);
    result->useForCompute();
    compute_shader->dispatch(glm::ivec3(ceil(dim.x / 8), ceil(dim.y / 8), 1));

    return result;
}

}    // namespace Noise
}    // namespace atcg