#include <Renderer/RenderGraph.h>

#include <Core/Assert.h>
#include <Renderer/RenderPasses/BlitPass.h>
#include <Renderer/RenderPasses/ForwardPass.h>
#include <Renderer/RenderPasses/ShadowPass.h>
#include <Renderer/RenderPasses/TonemapPass.h>
#include <Renderer/RenderPasses/DepthPass.h>
#include <Renderer/RenderPasses/OutlinePass.h>

namespace atcg
{

RenderGraph::RenderGraph()
{
    Dictionary output_pass_properties;
    _output_pass        = atcg::make_ref<OutputPass>(output_pass_properties);
    _output_pass_handle = addRenderPass(_output_pass);
}

RenderGraph::RenderPassHandle RenderGraph::addRenderPass(const atcg::ref_ptr<RenderPass>& pass)
{
    RenderPassHandle handle = (RenderPassHandle)_nodes.size();
    RenderPassNode node {handle, pass};
    _nodes.push_back(atcg::make_ref<RenderPassNode>(node));
    return handle;
}

void RenderGraph::addDependency(const RenderPassHandle& source,
                                std::string_view source_name,
                                const RenderPassHandle& target,
                                std::string_view target_name)
{
    _edges.push_back(PortEdge {source, target, std::string(source_name), std::string(target_name)});
}

void RenderGraph::compile(const CompileData& ctx)
{
    _compiled_passes.clear();
    _resource_map.clear();

    topologicalSort();

    reflect(ctx);

    _compiled     = true;
    _compile_data = ctx;
}

void RenderGraph::RenderGraph::topologicalSort()
{
    size_t node_size = _nodes.size();
    std::vector<int> inDegree(node_size, 0);
    std::vector<std::vector<RenderPassHandle>> adj(node_size);

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

        _compiled_passes.push_back(_nodes[handle]);

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
}

void RenderGraph::reflect(const CompileData& ctx)
{
    for(auto& node: _nodes)
    {
        node->reflection = node->pass->reflect(ctx);
    }
}

void RenderGraph::allocateResources(RenderGraphResources& resources, const RenderContext& ctx)
{
    resources.physical_resources.clear();
    resources.logicalToPhysicalResourceMap.clear();
    resources.render_pass_resource_tables.clear();

    auto output_fbo = _output_pass->outputFBO();

    if(!output_fbo)
    {
        ATCG_ERROR("Output pass does not have a valid output FBO set. Make sure to call setOutputFramebuffer() with a "
                   "valid FBO before executing the graph.");
        return;
    }

    // Create resource for each output
    for(const auto& node: _compiled_passes)
    {
        // Write relative texture sizes to outputs
        if(node->reflection.framebuffer_data.valid)
        {
            for(auto& handle: node->reflection.framebuffer_data.handles)
            {
                node->reflection.outputs[handle].desc.texture.width =
                    node->reflection.framebuffer_data.width.convert(output_fbo->width());
                node->reflection.outputs[handle].desc.texture.height =
                    node->reflection.framebuffer_data.height.convert(output_fbo->height());
            }
        }

        for(const auto& output: node->reflection.outputs)
        {
            std::string name                            = node->pass->name() + "." + output.name;
            auto& desc                                  = output.desc;
            RenderGraphResources::ResourceHandle handle = resources.physical_resources.size();
            resources.physical_resources.push_back(createResource(_compile_data, desc));
            resources.logicalToPhysicalResourceMap[name] = handle;
        }
    }


    for(const auto& edge: _edges)
    {
        std::string producerKey = _nodes[edge.from]->pass->name() + "." + edge.from_port;
        std::string consumerKey = _nodes[edge.to]->pass->name() + "." + edge.to_port;

        auto it = resources.logicalToPhysicalResourceMap.find(producerKey);
        if(it == resources.logicalToPhysicalResourceMap.end())
        {
            ATCG_ERROR("Producer resource {0} not found for edge from pass {1} to pass {2}",
                       producerKey,
                       _nodes[edge.from]->pass->name(),
                       _nodes[edge.to]->pass->name());
            continue;
        }

        resources.logicalToPhysicalResourceMap[consumerKey] = it->second;
    }
}

void RenderGraph::generateResourceTables(RenderGraphResources& resources, const RenderContext& ctx)
{
    auto output_fbo = _output_pass->outputFBO();

    if(!output_fbo)
    {
        ATCG_ERROR("Output pass does not have a valid output FBO set. Make sure to call setOutputFramebuffer() with a "
                   "valid FBO before executing the graph.");
        return;
    }

    // Generate resource tables
    for(auto node: _compiled_passes)
    {
        ResourceTable table;

        for(auto& input: node->reflection.inputs)
        {
            std::string key = node->pass->name() + "." + input.name;
            auto it         = resources.logicalToPhysicalResourceMap.find(key);
            if(it == resources.logicalToPhysicalResourceMap.end())
            {
                ATCG_WARN("No physical resource found for input {0} of pass {1}", input.name, node->pass->name());
                continue;
            }

            table.set(input.name, resources.physical_resources[it->second]);
        }

        if(node->reflection.framebuffer_data.valid)
        {
            uint32_t output_width  = node->reflection.framebuffer_data.width.convert(output_fbo->width());
            uint32_t output_height = node->reflection.framebuffer_data.height.convert(output_fbo->height());

            ATCG_ASSERT(output_width > 0 && output_height > 0,
                        "Failed to create framebuffer for pass" + node->pass->name());

            atcg::ref_ptr<Framebuffer> target_fbo = atcg::make_ref<Framebuffer>(output_width, output_height);

            for(auto& handle: node->reflection.framebuffer_data.handles)
            {
                const auto& output = node->reflection.outputs[handle];
                std::string key    = node->pass->name() + "." + output.name;
                auto resource      = resources.physical_resources[resources.logicalToPhysicalResourceMap[key]];
                table.set(output.name, resource);

                if(target_fbo == nullptr)
                {
                    ATCG_ERROR("Output {0} of pass {1} is a framebuffer attachement but no target FBO was created",
                               output.name,
                               node->pass->name());
                    continue;
                }

                auto texture = std::get<atcg::ref_ptr<Texture>>(resource);

                if(output.desc.texture.format == TextureFormat::DEPTH)
                {
                    target_fbo->attachDepth(texture);
                }
                else
                {
                    target_fbo->attachTexture(texture);
                }
            }

            target_fbo->complete();
            table.setTargetFBO(target_fbo);
        }

        resources.render_pass_resource_tables.insert(std::make_pair(node->handle, table));
    }
}

RenderGraph::RenderGraphResources& RenderGraph::getResources(const RenderContext& ctx)
{
    auto output_fbo = _output_pass->outputFBO();

    FramebufferResolution resolution {output_fbo->width(), output_fbo->height()};
    auto it = _resource_map.find(resolution);
    if(it == _resource_map.end())
    {
        if(_resource_map.size() > _max_cached_resolutions)
        {
            ATCG_WARN("More than {0} different framebuffer resolutions detected in render graph. Cache is cleared. "
                      "This setting can be changed with render_graph->setMaxCachedResolutions()",
                      _max_cached_resolutions);
            _resource_map.clear();
        }

        RenderGraphResources new_resources;

        allocateResources(new_resources, ctx);
        generateResourceTables(new_resources, ctx);

        _resource_map[resolution] = new_resources;
        it                        = _resource_map.find(resolution);
    }
    return it->second;
}

void RenderGraph::execute(const RenderContext& ctx)
{
    auto& resources = getResources(ctx);

    for(auto node: _compiled_passes)
    {
        node->pass->execute(ctx, resources.render_pass_resource_tables.at(node->handle));
    }
}

void RenderGraph::exportToDOT(const std::string& path) const
{
    std::ofstream out(path);
    out << "digraph RenderGraph {\n";
    out << "    rankdir=LR;\n";    // optional: makes the graph left-to-right instead of top-down

    for(RenderPassHandle i = 0; i < _nodes.size(); ++i)
    {
        out << "    " << i << " [label=\"" << _nodes[i]->pass->name() << "\"];\n";
    }

    for(const auto& edge: _edges)
    {
        out << "    " << edge.from << " -> " << edge.to << " [label=\"" << edge.from_port << " → " << edge.to_port
            << "\"];\n";
    }

    out << "}\n";
}

atcg::ref_ptr<RenderGraph> createRenderGraph(const CompileData& ctx)
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

}    // namespace atcg