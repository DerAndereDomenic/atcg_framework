#pragma once

#include <Core/API.h>
#include <Renderer/RenderPass.h>
#include <DataStructure/Dictionary.h>
#include <Renderer/CompileData.h>
#include <Renderer/RenderPasses/OutputPass.h>

#include <queue>

namespace atcg
{

/**
 * @brief A class the model a RenderGraph.
 */
class ATCG_API RenderGraph
{
public:
    using RenderPassHandle = uint64_t;

    /**
     * @brief Create a Rendergraph
     */
    RenderGraph();

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
    void compile(const CompileData& ctx);

    /**
     * @brief Execute the graph.
     *
     * @param ctx The context holding per-frame data
     */
    void execute(const RenderContext& ctx);

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
     * @brief Compile the graph if it is not compiled, otherwise NOP
     *
     * @param ctx The compile context
     */
    ATCG_INLINE void ensureCompiled(const CompileData& ctx)
    {
        if(!_compiled) compile(ctx);
    }

    /**
     * @brief Get the output pass of the graph
     *
     * @return The output pass of the graph
     */
    ATCG_INLINE const atcg::ref_ptr<OutputPass>& outputPass() const { return _output_pass; }

    /**
     * @brief Get the output pass handle of the graph
     *
     * @return The output pass handle of the graph
     */
    ATCG_INLINE RenderPassHandle outputPassHandle() const { return _output_pass_handle; }

    /**
     * @brief Set the output framebuffer of the output pass
     *
     * @param fbo The output framebuffer
     */
    ATCG_INLINE void setOutputFramebuffer(const atcg::ref_ptr<Framebuffer>& fbo) { _output_pass->setOutputFBO(fbo); }

    /**
     * @brief Set the maximum number of different framebuffer resolutions that are cached by the render graph. If more
     * different framebuffer resolutions are detected, the cache is cleared. This is used to prevent the render graph
     * from consuming too much memory if the output resolution changes frequently (e.g. when resizing the window).
     *
     * @param max_cached_resolutions The maximum number of different framebuffer resolutions that are cached by the
     * render graph
     */
    ATCG_INLINE void setMaxCachedResolutions(uint32_t max_cached_resolutions)
    {
        _max_cached_resolutions = max_cached_resolutions;
    }

private:
    struct RenderPassNode
    {
        RenderPassHandle handle;
        atcg::ref_ptr<RenderPass> pass;
        RenderPassReflection reflection;
    };

    struct PortEdge
    {
        RenderPassHandle from;
        RenderPassHandle to;
        std::string from_port;
        std::string to_port;
    };

    struct FramebufferResolution
    {
        uint32_t width;
        uint32_t height;

        bool operator==(const FramebufferResolution& other) const noexcept
        {
            return width == other.width && height == other.height;
        }
    };

    struct FramebufferResolutionHash
    {
        size_t operator()(const FramebufferResolution& r) const noexcept
        {
            size_t h1 = std::hash<uint32_t> {}(r.width);
            size_t h2 = std::hash<uint32_t> {}(r.height);
            return h1 ^ (h2 << 1);
        }
    };

    struct RenderGraphResources
    {
        using ResourceHandle = uint32_t;
        std::unordered_map<std::string, ResourceHandle> logicalToPhysicalResourceMap;
        std::vector<PhysicalResource> physical_resources;
        std::unordered_map<RenderPassHandle, ResourceTable> render_pass_resource_tables;
    };

    using ResourceMap = std::unordered_map<FramebufferResolution, RenderGraphResources, FramebufferResolutionHash>;

private:
    void topologicalSort();

    void reflect(const CompileData& ctx);

    void allocateResources(RenderGraphResources& resources, const RenderContext& ctx);

    void generateResourceTables(RenderGraphResources& resources, const RenderContext& ctx);

    RenderGraphResources& getResources(const RenderContext& ctx);

private:
    std::vector<atcg::ref_ptr<RenderPassNode>> _nodes;
    std::vector<atcg::ref_ptr<RenderPassNode>> _compiled_passes;    // Same data as _passes but topologically sorted

    std::vector<PortEdge> _edges;

    ResourceMap _resource_map;

    CompileData _compile_data;
    bool _compiled = false;

    atcg::ref_ptr<OutputPass> _output_pass;
    uint32_t _output_pass_handle;

    uint32_t _max_cached_resolutions = 8;
};
}    // namespace atcg