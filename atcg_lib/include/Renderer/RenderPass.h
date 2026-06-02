#pragma once

#include <Core/Platform.h>
#include <Core/Memory.h>
#include <DataStructure/Dictionary.h>
#include <Renderer/Framebuffer.h>
#include <Renderer/RenderPassReflection.h>
#include <Renderer/ResourceTable.h>
#include <Renderer/RenderContext.h>
#include <Renderer/CompileData.h>

#include <any>

namespace atcg
{

/**
 * @brief A class to model a render pass
 *
 * The output type of the RenderPass
 */
class RenderPass
{
public:
    /**
     * @brief Default constructor
     *
     * @param properties A dictionary of properties that can be used to configure the render pass
     * @param name The name of the render pass (has to be unique within a render graph)
     */
    RenderPass(Dictionary& properties, std::string_view name = "RenderPass") : _name(name) {}

    /**
     * @brief Destructor
     */
    virtual ~RenderPass() = default;

    /**
     * @brief Reflect the render pass. This function describes the inputs, outputs and framebuffer data of this render
     * pass. This is used by the render graph to generate the resource tables and framebuffers for this render pass.
     *
     * @param ctx The compile data
     * @return The reflection data of this render pass
     */
    virtual RenderPassReflection reflect(const CompileData& ctx) = 0;

    /**
     * @brief Execute the render pass. This function is called by the render graph to execute this render pass. The
     * resources used by this render pass are passed in the resource table.
     *
     * @param ctx The render context holding per-frame data
     * @param resources The resource table holding the resources for this render pass
     */
    virtual void execute(const RenderContext& ctx, const ResourceTable& resources) = 0;

    /**
     * @brief Get the name of the render pass
     *
     * @return The name of the render pass
     */
    ATCG_INLINE const std::string& name() const { return _name; }

protected:
    std::string _name;
};

}    // namespace atcg