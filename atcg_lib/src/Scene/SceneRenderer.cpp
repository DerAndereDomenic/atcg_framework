#include <Scene/SceneRenderer.h>

namespace atcg
{
class SceneRendererSystem::Impl
{
public:
    Impl();

    ~Impl();

    RendererSystem* _renderer                = nullptr;
    atcg::ref_ptr<RenderGraph> _render_graph = nullptr;
};

SceneRendererSystem::Impl::Impl() {}

SceneRendererSystem::Impl::~Impl() {}

SceneRendererSystem::SceneRendererSystem(RendererSystem* renderer)
{
    impl            = std::make_unique<Impl>();
    impl->_renderer = renderer;

    CompileData data;
    data.num_samples    = 1;
    impl->_render_graph = createRenderGraph(data);
}

SceneRendererSystem::~SceneRendererSystem() {}

void SceneRendererSystem::render(const atcg::ref_ptr<Scene>& scene,
                                 const atcg::ref_ptr<Camera>& camera,
                                 const atcg::ref_ptr<Framebuffer>& target_fbo,
                                 bool draw_cameras)
{
    RenderContext ctx;
    ctx.renderer     = impl->_renderer;
    ctx.scene        = scene;
    ctx.camera       = camera;
    ctx.draw_cameras = draw_cameras;

    impl->_render_graph->setOutputFramebuffer(target_fbo);

    impl->_render_graph->execute(ctx);
}

atcg::ref_ptr<RenderGraph> SceneRendererSystem::getRenderGraph() const
{
    return impl->_render_graph;
}

void SceneRendererSystem::setRenderGraph(const atcg::ref_ptr<RenderGraph>& graph)
{
    impl->_render_graph = graph;
}

}    // namespace atcg