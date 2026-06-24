#include <Scene/SceneRenderer.h>

#include <Renderer/RenderPasses/BlitPass.h>
#include <Renderer/RenderPasses/ForwardPass.h>
#include <Renderer/RenderPasses/ShadowPass.h>
#include <Renderer/RenderPasses/TonemapPass.h>
#include <Renderer/RenderPasses/DepthPass.h>
#include <Renderer/RenderPasses/OutlinePass.h>

namespace atcg
{
class SceneRendererSystem::Impl
{
public:
    Impl();

    ~Impl();

    atcg::ref_ptr<RenderGraph> createRenderGraph();

    uint32_t num_msaa_samples       = 1;
    uint32_t shadow_pass_resolution = 1024;

    RendererSystem* _renderer                = nullptr;
    atcg::ref_ptr<RenderGraph> _render_graph = nullptr;
};

SceneRendererSystem::Impl::Impl() {}

SceneRendererSystem::Impl::~Impl() {}

atcg::ref_ptr<RenderGraph> SceneRendererSystem::Impl::createRenderGraph()
{
    auto graph = atcg::make_ref<RenderGraph>();

    Dictionary forward_pass_properties;
    atcg::ref_ptr<ForwardPass> forward_pass = atcg::make_ref<ForwardPass>(forward_pass_properties);
    Dictionary tonemap_pass_properties;
    atcg::ref_ptr<TonemapPass> tonemap_pass = atcg::make_ref<TonemapPass>(tonemap_pass_properties);
    Dictionary depth_pass_properties;
    depth_pass_properties.setValue("cull_mode", CullMode::ATCG_FRONT_FACE_CULLING);
    atcg::ref_ptr<DepthPass> depth_pass = atcg::make_ref<DepthPass>(depth_pass_properties);
    Dictionary shadow_pass_properties;
    shadow_pass_properties.setValue("resolution", shadow_pass_resolution);
    atcg::ref_ptr<ShadowPass> shadow_pass = atcg::make_ref<ShadowPass>(shadow_pass_properties);
    Dictionary outline_pass_properties;
    atcg::ref_ptr<OutlinePass> outline_pass = atcg::make_ref<OutlinePass>(outline_pass_properties);

    auto forward_handle = graph->addRenderPass(forward_pass);
    auto tonemap_handle = graph->addRenderPass(tonemap_pass);
    auto depth_handle   = graph->addRenderPass(depth_pass);
    auto shadow_handle  = graph->addRenderPass(shadow_pass);
    auto outline_handle = graph->addRenderPass(outline_pass);
    auto output_handle  = graph->outputPassHandle();

    graph->addDependency(depth_handle, "depth_buffer", forward_handle, "depth_buffer");
    graph->addDependency(shadow_handle, "point_light_depth_maps", forward_handle, "point_light_depth_maps");

    CompileData ctx;
    ctx.num_samples = num_msaa_samples;

    if(ctx.num_samples > 1)
    {
        Dictionary blit_pass_properties;
        atcg::ref_ptr<BlitPass> blit_pass = atcg::make_ref<BlitPass>(blit_pass_properties);

        auto blit_handle = graph->addRenderPass(blit_pass);

        graph->addDependency(forward_handle, "output", blit_handle, "input_color_buffer");
        graph->addDependency(forward_handle, "out_depth_buffer", blit_handle, "input_depth_buffer");
        graph->addDependency(forward_handle, "entity_buffer", blit_handle, "input_entity_buffer");
        graph->addDependency(forward_handle, "stencil_buffer", blit_handle, "input_stencil_buffer");

        graph->addDependency(blit_handle, "out_color_buffer", tonemap_handle, "hdr");
        graph->addDependency(blit_handle, "out_stencil_buffer", tonemap_handle, "in_stencil_buffer");
        graph->addDependency(blit_handle, "out_entity_buffer", output_handle, "entities");
        graph->addDependency(blit_handle, "out_stencil_buffer", output_handle, "stencil");
        graph->addDependency(blit_handle, "out_depth_buffer", output_handle, "depth");
        graph->addDependency(blit_handle, "out_entity_buffer", outline_handle, "input_entity_buffer");
    }
    else
    {
        graph->addDependency(forward_handle, "output", tonemap_handle, "hdr");
        graph->addDependency(forward_handle, "stencil_buffer", tonemap_handle, "in_stencil_buffer");
        graph->addDependency(forward_handle, "entity_buffer", output_handle, "entities");
        graph->addDependency(forward_handle, "stencil_buffer", output_handle, "stencil");
        graph->addDependency(forward_handle, "out_depth_buffer", output_handle, "depth");
        graph->addDependency(forward_handle, "entity_buffer", outline_handle, "input_entity_buffer");
    }

    graph->addDependency(tonemap_handle, "output_color", outline_handle, "input_color_buffer");
    graph->addDependency(outline_handle, "out_color_buffer", output_handle, "color");

    graph->compile(ctx);

    return graph;
}

SceneRendererSystem::SceneRendererSystem(RendererSystem* renderer)
{
    impl            = std::make_unique<Impl>();
    impl->_renderer = renderer;

    impl->_render_graph = impl->createRenderGraph();
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

void SceneRendererSystem::setNumberMSAASamples(uint32_t num_samples)
{
    impl->num_msaa_samples = num_samples;
    impl->_render_graph    = impl->createRenderGraph();
}

void SceneRendererSystem::setShadowPassResolution(uint32_t resolution)
{
    impl->shadow_pass_resolution = resolution;
    impl->_render_graph          = impl->createRenderGraph();
}

}    // namespace atcg