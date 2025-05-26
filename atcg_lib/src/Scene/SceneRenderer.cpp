#include <Scene/SceneRenderer.h>

#include <Core/Assert.h>
#include <Scene/Components.h>
#include <Scene/Entity.h>
#include <Scene/ComponentRenderer.h>
#include <Renderer/RenderGraph.h>

#include <Renderer/RenderPasses/SkyboxPass.h>
#include <Renderer/RenderPasses/ShadowPass.h>
#include <Renderer/RenderPasses/ForwardPass.h>

namespace atcg
{

class SceneRenderer::Impl
{
public:
    Impl();

    ~Impl();

    Dictionary context;

    atcg::ref_ptr<atcg::RenderGraph> graph;
};

SceneRenderer::Impl::Impl()
{
    graph = atcg::make_ref<atcg::RenderGraph>();

    auto skybox_handle = graph->addRenderPass(atcg::make_ref<SkyboxPass>());
    auto shadow_handle = graph->addRenderPass(atcg::make_ref<ShadowPass>());
    auto output_handle = graph->addRenderPass(atcg::make_ref<ForwardPass>());

    graph->addDependency(skybox_handle, "framebuffer", output_handle, "framebuffer");
    graph->addDependency(shadow_handle, "point_light_depth_maps", output_handle, "point_light_depth_maps");

    graph->compile(context);
}

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
    impl->context.setValue("scene", scene);
}

void SceneRenderer::render(const atcg::ref_ptr<Camera>& camera)
{
    impl->context.setValue("camera", camera);
    impl->graph->execute(impl->context);
}
}    // namespace atcg