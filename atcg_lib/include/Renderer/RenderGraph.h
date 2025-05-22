#pragma once

#include <Renderer/RenderPass.h>
#include <DataStructure/Dictionary.h>

#include <queue>

namespace atcg
{

/**
 * @brief A class the model a RenderGraph.
 *
 * @tparam RenderContextT The Render context data
 */
class RenderGraph
{
public:
    using RenderPassHandle = uint64_t;

    /**
     * @brief Create a Rendergraph
     */
    RenderGraph() = default;

    /**
     * @brief Add a render pass to the graph.
     * This functions returns a handle and a RenderPassBuilder. The builder is an intermediate class that collects all
     * the data associated with the rendering pass and then gets compiled into the final render pass when compile() is
     * called. The handle can be used to access different render passes to add dependencies between them (by using
     * addDependency()).
     *
     * @return A tuple with a RenderPassHandle and a RenderPassBuilder
     */
    std::pair<RenderPassHandle, atcg::ref_ptr<RenderPassBuilder>> addRenderPass()
    {
        auto builder            = atcg::make_ref<RenderPassBuilder>();
        RenderPassHandle handle = (RenderPassHandle)_builder.size();
        _builder.push_back(builder);
        return std::make_pair(handle, builder);
    }

    /**
     * @brief Create a dependency between two Renderpasses.
     * Creates the directed edge (source, target) into the Graph. The handles are obtained by calling addRenderPass().
     *
     * @param source The source handle
     * @param target The target handle
     */
    void addDependency(const RenderPassHandle& source,
                       std::string_view source_name,
                       const RenderPassHandle& target,
                       std::string_view target_name)
    {
        _edges.push_back(PortEdge {source, target, std::string(source_name), std::string(target_name)});
    }

    /**
     * @brief Compile the graph.
     * This has to be called before executing the graph.
     */
    void compile(Dictionary& ctx)
    {
        size_t node_size = _builder.size();
        std::vector<int> inDegree(node_size, 0);
        std::vector<std::vector<RenderPassHandle>> adj(node_size);

        std::vector<atcg::ref_ptr<RenderPass>> passes;
        for(auto builder: _builder)
        {
            auto pass = builder->build(ctx);
            passes.push_back(pass);
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

            const auto& outputs = passes[from]->getOutputs();
            if(!outputs.contains(edge.from_port))
            {
                ATCG_ERROR("Error while compiling Render Graph: {} does not exist as an output", edge.from_port);
                continue;
            }

            passes[to]->addInput(edge.to_port, outputs.getValueRaw(edge.from_port));
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

            _compiled_passes.push_back(passes[handle]);

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

        _builder.clear();
    }

    /**
     * @brief Execute the graph.
     */
    void execute(Dictionary& ctx)
    {
        for(auto pass: _compiled_passes)
        {
            pass->execute(ctx);
        }
    }


private:
    struct PortEdge
    {
        RenderPassHandle from;
        RenderPassHandle to;
        std::string from_port;
        std::string to_port;
    };

private:
    std::vector<atcg::ref_ptr<RenderPassBuilder>> _builder;
    std::vector<atcg::ref_ptr<RenderPass>> _compiled_passes;
    std::vector<PortEdge> _edges;
};
}    // namespace atcg