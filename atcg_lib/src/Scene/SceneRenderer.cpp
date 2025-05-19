#include <Scene/SceneRenderer.h>

#include <Core/Assert.h>
#include <Renderer/Renderer.h>

namespace atcg
{
class SceneRenderer::Impl
{
public:
    Impl();

    ~Impl();

    atcg::ref_ptr<atcg::Scene> scene;
};

SceneRenderer::Impl::Impl() {}

SceneRenderer::Impl::~Impl() {}

SceneRenderer::SceneRenderer()
{
    impl = std::make_unique<Impl>();
}

SceneRenderer::SceneRenderer(const atcg::ref_ptr<atcg::Scene>& scene)
{
    impl = std::make_unique<Impl>();
    setScene(scene);
}

SceneRenderer::~SceneRenderer() {}

void SceneRenderer::setScene(const atcg::ref_ptr<atcg::Scene>& scene)
{
    impl->scene = scene;
}

void SceneRenderer::render(const atcg::ref_ptr<Camera>& camera)
{
    ATCG_ASSERT(impl->scene, "No scene set");

    atcg::Renderer::draw(impl->scene, camera);
}
}    // namespace atcg