#pragma once

#include <Renderer/RenderPass.h>
#include <DataStructure/Dictionary.h>

#include <queue>

namespace atcg
{

/**
 * @brief A class the model a RenderGraph.
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
     * This functions returns a handle and a RenderPass. The handle can be used to access different render passes to add
     * dependencies between them (by using addDependency()).
     *
     * @param name The name of the RenderPass
     *
     * @return A tuple with a RenderPassHandle and a RenderPass
     */
    std::pair<RenderPassHandle, atcg::ref_ptr<RenderPass>> addRenderPass(std::string_view name = "")
    {
        auto builder            = atcg::make_ref<RenderPass>(name);
        RenderPassHandle handle = (RenderPassHandle)_passes.size();
        _passes.push_back(builder);
        return std::make_pair(handle, builder);
    }

    /**
     * @brief Add a render pass to the graph.
     * The handle can be used to access different render passes to add dependencies between them (by using
     * addDependency()).
     *
     * @param pass The render pass
     *
     * @return The handle to the renderpass
     */
    RenderPassHandle addRenderPass(const atcg::ref_ptr<RenderPass>& pass)
    {
        RenderPassHandle handle = (RenderPassHandle)_passes.size();
        _passes.push_back(pass);
        return handle;
    }

    /**
     * @brief Create a connection between two ports of an output and input node.
     * The handles are obtained by "addRenderPass()"
     *
     * @param source The source handle
     * @param source_name The name of the source port
     * @param target The target handle
     * @param target_name The name of the target port
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
     *
     * @param ctx The context
     */
    void compile(Dictionary& ctx)
    {
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
    }

    /**
     * @brief Execute the graph.
     *
     * @param ctx The context holding per-frame data
     */
    void execute(Dictionary& ctx)
    {
        for(auto pass: _compiled_passes)
        {
            pass->execute(ctx);
        }
    }

    /**
     * @brief Exports the graph into a DOT format txt file for debugging porpuses.
     *
     * @param path The path of the exported file
     */
    void exportToDOT(const std::string& path) const
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


private:
    struct PortEdge
    {
        RenderPassHandle from;
        RenderPassHandle to;
        std::string from_port;
        std::string to_port;
    };

private:
    std::vector<atcg::ref_ptr<RenderPass>> _passes;
    std::vector<atcg::ref_ptr<RenderPass>> _compiled_passes;    // Same data as _passes but topologically sorted
    std::vector<PortEdge> _edges;
};
}    // namespace atcg