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

}    // namespace Noise
}    // namespace atcg