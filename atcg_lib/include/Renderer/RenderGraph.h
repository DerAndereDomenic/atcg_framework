#pragma once

#include <Renderer/RenderPass.h>

#include <queue>

namespace atcg
{

/**
 * @brief A class the model a RenderGraph.
 *
 * @tparam RenderContextT The Render context data
 */
template<typename RenderContextT>
class RenderGraph
{
public:
    using RenderPassHandle = uint64_t;

    /**
     * @brief Create a Rendergraph
     *
     * @param ctx The Render context
     */
    RenderGraph(const atcg::ref_ptr<RenderContextT>& ctx) : _context(ctx) {}

    /**
     * @brief Add a render pass to the graph.
     * This functions returns a handle and a RenderPassBuilder. The builder is an intermediate class that collects all
     * the data associated with the rendering pass and then gets compiled into the final render pass when compile() is
     * called. The handle can be ussed to access different render passes to add dependencies between them (by using
     * addDependency()).
     *
     * @tparam RenderPassDataT The datatype of intermediate buffers used during the render pass
     * @tparam RenderPassOutputT The output type of the RenderPass
     *
     * @return A tuple with a RenderPassHandle and a RenderPassBuilder
     */
    template<typename RenderPassDataT, typename RenderPassOutputT>
    std::pair<RenderPassHandle, atcg::ref_ptr<RenderPassBuilder<RenderContextT, RenderPassDataT, RenderPassOutputT>>>
    addRenderPass()
    {
        auto builder = atcg::make_ref<RenderPassBuilder<RenderContextT, RenderPassDataT, RenderPassOutputT>>();
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
    void addDependency(const RenderPassHandle& source, const RenderPassHandle& target)
    {
        _edges.push_back(std::make_pair(source, target));
    }

    /**
     * @brief Compile the graph.
     * This has to be called before executing the graph.
     */
    void compile()
    {
        size_t node_size = _builder.size();
        std::vector<int> inDegree(node_size, 0);
        std::vector<std::vector<RenderPassHandle>> adj(node_size);

        std::vector<atcg::ref_ptr<RenderPassBase<RenderContextT>>> passes;
        for(auto builder: _builder)
        {
            auto pass = builder->build(_context);
            passes.push_back(pass);
        }

        for(const auto& [from, to]: _edges)
        {
            adj[from].push_back(to);
            ++inDegree[to];
            passes[to]->addInput(passes[from]->getOutput());
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
    void execute()
    {
        for(auto pass: _compiled_passes)
        {
            pass->execute(_context);
        }
    }

private:
    std::vector<atcg::ref_ptr<RenderPassBuilderBase<RenderContextT>>> _builder;
    std::vector<atcg::ref_ptr<RenderPassBase<RenderContextT>>> _compiled_passes;
    std::vector<std::tuple<RenderPassHandle, RenderPassHandle>> _edges;
    atcg::ref_ptr<RenderContextT> _context;
};
}    // namespace atcg