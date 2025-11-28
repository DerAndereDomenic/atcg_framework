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
    std::pair<RenderPassHandle, atcg::ref_ptr<RenderPass>> addRenderPass(std::string_view name = "");

    /**
     * @brief Add a render pass to the graph.
     * This functions returns a handle and a RenderPass. The handle can be used to access different render passes to add
     * dependencies between them (by using addDependency()).
     *
     * @param desc The Render target description
     * @param name The name of the RenderPass
     *
     * @return A tuple with a RenderPassHandle and a RenderPass
     */
    std::pair<RenderPassHandle, atcg::ref_ptr<RenderPass>> addRenderPass(const RenderTargetDesc& desc,
                                                                         std::string_view name = "");

    /**
     * @brief Add a render pass to the graph.
     * The handle can be used to access different render passes to add dependencies between them (by using
     * addDependency()).
     *
     * @param pass The render pass
     *
     * @return The handle to the renderpass
     */
    RenderPassHandle addRenderPass(const atcg::ref_ptr<RenderPass>& pass);

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
                       std::string_view target_name);

    /**
     * @brief Compile the graph.
     * This has to be called before executing the graph.
     *
     * @param ctx The context
     */
    void compile(Dictionary& ctx);

    /**
     * @brief Execute the graph.
     *
     * @param ctx The context holding per-frame data
     */
    void execute(Dictionary& ctx);

    /**
     * @brief Exports the graph into a DOT format txt file for debugging porpuses.
     *
     * @param path The path of the exported file
     */
    void exportToDOT(const std::string& path) const;

    /**
     * @brief Check if the model is compiled
     *
     * @return If the graph is compiled
     */
    ATCG_INLINE bool isCompiled() const { return _compiled; }

    /**
     * @brief Trigger the garbage collection for each render pass.
     * This function increases the lifetime of garbage collected objects and destroys them if the maximum life time is
     * reached.
     */
    void garbageCollect();

    /**
     * @brief Compile the graph if it is not compiled, otherwise NOP
     *
     * @param ctx The compile context
     */
    ATCG_INLINE void ensureCompiled(Dictionary& ctx)
    {
        if(!_compiled) compile(ctx);
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

    bool _compiled = false;
};

atcg::ref_ptr<RenderGraph> createStandardGraph();

atcg::ref_ptr<RenderGraph> createMSAAGraph(uint32_t num_samples);
}    // namespace atcg