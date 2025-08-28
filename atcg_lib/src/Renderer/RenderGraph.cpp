#include <Renderer/RenderGraph.h>

#include <Renderer/RenderPasses/BlitPass.h>
#include <Renderer/RenderPasses/SkyboxPass.h>
#include <Renderer/RenderPasses/ForwardPass.h>
#include <Renderer/RenderPasses/ShadowPass.h>

namespace atcg
{
std::pair<RenderGraph::RenderPassHandle, atcg::ref_ptr<RenderPass>>
RenderGraph::addRenderPass(const RenderTargetDesc& desc, std::string_view name)
{
    auto builder            = atcg::make_ref<RenderPass>(desc, name);
    RenderPassHandle handle = (RenderPassHandle)_passes.size();
    _passes.push_back(builder);
    return std::make_pair(handle, builder);
}

RenderGraph::RenderPassHandle RenderGraph::addRenderPass(const atcg::ref_ptr<RenderPass>& pass)
{
    RenderPassHandle handle = (RenderPassHandle)_passes.size();
    _passes.push_back(pass);
    return handle;
}

void RenderGraph::addDependency(const RenderPassHandle& source,
                                std::string_view source_name,
                                const RenderPassHandle& target,
                                std::string_view target_name)
{
    _edges.push_back(PortEdge {source, target, std::string(source_name), std::string(target_name)});
}

void RenderGraph::compile(Dictionary& ctx)
{
    _compiled_passes.clear();

    size_t node_size = _passes.size();
    std::vector<int> inDegree(node_size, 0);
    std::vector<std::vector<RenderPassHandle>> adj(node_size);

    for(auto pass: _passes)
    {
        pass->setup(ctx);
    }

    std::set<std::pair<RenderPassHandle, RenderPassHandle>> uniqueEdges;
    for(const auto& edge: _edges)
    {
        RenderPassHandle from = edge.from;
        RenderPassHandle to   = edge.to;

        if(uniqueEdges.insert({from, to}).second)
        {
            adj[from].push_back(to);
            ++inDegree[to];
        }

        const auto& outputs = _passes[from]->getOutputs();
        if(!outputs.contains(edge.from_port))
        {
            ATCG_ERROR("Error while compiling Render Graph: {} does not exist as an output", edge.from_port);
            continue;
        }

        _passes[to]->addInput(edge.to_port, outputs.getValueRaw(edge.from_port));
    }

    std::queue<RenderPassHandle> zeroInDegree;
    for(RenderPassHandle i = 0; i < node_size; ++i)
    {
        if(inDegree[i] == 0)
        {
            zeroInDegree.push(i);
        }
    }

    while(!zeroInDegree.empty())
    {
        RenderPassHandle handle = zeroInDegree.front();
        zeroInDegree.pop();

        _compiled_passes.push_back(_passes[handle]);

        for(int neighbor: adj[handle])
        {
            if(--inDegree[neighbor] == 0)
            {
                zeroInDegree.push(neighbor);
            }
        }
    }

    if(_compiled_passes.size() != node_size)
    {
        throw std::runtime_error("Graph has a cycle. Topological sorting is not possible.");
    }

    _compiled = true;
}

void RenderGraph::execute(Dictionary& ctx)
{
    for(auto pass: _compiled_passes)
    {
        pass->execute(ctx);
    }
}

void RenderGraph::exportToDOT(const std::string& path) const
{
    std::ofstream out(path);
    out << "digraph RenderGraph {\n";
    out << "    rankdir=LR;\n";    // optional: makes the graph left-to-right instead of top-down

    for(RenderPassHandle i = 0; i < _passes.size(); ++i)
    {
        out << "    " << i << " [label=\"" << _passes[i]->name() << "\"];\n";
    }

    for(const auto& edge: _edges)
    {
        out << "    " << edge.from << " -> " << edge.to << " [label=\"" << edge.from_port << " → " << edge.to_port
            << "\"];\n";
    }

    out << "}\n";
}


atcg::ref_ptr<RenderGraph> createStandardGraph()
{
    auto _render_graph = atcg::make_ref<atcg::RenderGraph>();

    RenderTargetDesc render_desc;
    render_desc.clear = true;

    auto skybox_handle  = _render_graph->addRenderPass(atcg::make_ref<SkyboxPass>(render_desc));
    auto shadow_handle  = _render_graph->addRenderPass(atcg::make_ref<ShadowPass>());
    auto forward_handle = _render_graph->addRenderPass(atcg::make_ref<ForwardPass>());

    _render_graph->addDependency(skybox_handle, "skybox", forward_handle, "skybox");
    _render_graph->addDependency(skybox_handle, "framebuffer", forward_handle, "framebuffer");
    _render_graph->addDependency(shadow_handle, "point_light_depth_maps", forward_handle, "point_light_depth_maps");

    return _render_graph;
}

atcg::ref_ptr<RenderGraph> createMSAAGraph(uint32_t num_samples)
{
    auto _render_graph = atcg::make_ref<atcg::RenderGraph>();

    RenderTargetDesc render_desc;
    render_desc.mode        = RenderTargetMode::RENDER_TARGET_OWN_FRAMEBUFFER;
    render_desc.target_spec = FramebufferSpecification(
        1,
        1,
        num_samples,
        {
            {TextureFormat::RGBA, FramebufferTextureFormat::TEXTURE_2D_MULTISAMPLE},          // Color
            {TextureFormat::RINT, FramebufferTextureFormat::TEXTURE_2D_MULTISAMPLE},          // Entity ids
            {TextureFormat::DEPTH, FramebufferTextureFormat::TEXTURE_2D_MULTISAMPLE, true}    // Depth
        });
    render_desc.clear = true;

    auto skybox_handle  = _render_graph->addRenderPass(atcg::make_ref<SkyboxPass>(render_desc));
    auto shadow_handle  = _render_graph->addRenderPass(atcg::make_ref<ShadowPass>());
    auto forward_handle = _render_graph->addRenderPass(
        atcg::make_ref<ForwardPass>(RenderTargetDesc(RenderTargetMode::RENDER_TARGET_INPUT_FRAMEBUFFER)));
    auto screen_handle = _render_graph->addRenderPass(atcg::make_ref<BlitPass>());

    _render_graph->addDependency(skybox_handle, "skybox", forward_handle, "skybox");
    _render_graph->addDependency(skybox_handle, "framebuffer", forward_handle, "framebuffer");
    _render_graph->addDependency(shadow_handle, "point_light_depth_maps", forward_handle, "point_light_depth_maps");
    _render_graph->addDependency(forward_handle, "framebuffer", screen_handle, "framebuffer");

    return _render_graph;
}

}    // namespace atcg