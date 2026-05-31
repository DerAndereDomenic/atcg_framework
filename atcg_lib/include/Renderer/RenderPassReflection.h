#pragma once

#include <Core/API.h>
#include <DataStructure/GPUResource.h>

namespace atcg
{

/**
 * @brief A class to model the reflection data of a render pass. This is used by the render graph to generate the
 * resource tables and framebuffers for this render pass.
 *
 * @note No input - output validation is done. Therefore, no resource descriptions are required for the inputs.
 * For textures, a output size can be specified explicitely. However, if the handle of a texture input is passed to
 * setOutputFramebufferData, it is overwritten with the size of the framebuffer.
 */
struct ATCG_API RenderPassReflection
{
    RenderPassReflection() = default;

    struct LogicalResource
    {
        std::string name;
        ResourceDescription desc;
    };

    struct OutputFramebufferData
    {
        bool valid = false;
        TextureSize width;
        TextureSize height;
        std::vector<uint32_t> handles;
    };

    std::vector<LogicalResource> inputs;
    std::vector<LogicalResource> outputs;
    OutputFramebufferData framebuffer_data;

    /**
     * @brief Add an input to the render pass reflection. The name of the input has to be unique within this render
     * pass.
     *
     * @param name The name of the input
     */
    void addInput(std::string_view name)
    {
        inputs.push_back(LogicalResource {std::string(name), ResourceDescription()});
    }

    /**
     * @brief Add an output to the render pass reflection. The name of the output has to be unique within this render
     * pass.
     *
     * @param name The name of the output
     * @param desc The description of the output resource
     *
     * @return The index of the added output (used to specify the output framebuffer)
     */
    uint32_t addOutput(std::string_view name, const ResourceDescription& desc)
    {
        outputs.push_back(LogicalResource {std::string(name), desc});
        return outputs.size() - 1;
    }

    /**
     * @brief Set the output framebuffer data for this render pass. This specifies that the render pass should render to
     * a framebuffer with the specified width and height. The handles specify which outputs of this render pass should
     * be used as framebuffer attachments. The size of the framebuffer is determined by the width and height parameters.
     * If the framebuffer data is valid, the render graph creates a framebuffer for this render pass and binds the
     * specified outputs to this framebuffer. If the framebuffer data is not valid, no framebuffer is created for this
     * render pass and the outputs are not bound to a framebuffer.
     *
     * @note If the handle of a texture is passed to setOutputFramebufferData, its size is overwritten with the size of
     * the framebuffer.
     *
     * @param width The width of the output framebuffer. This can be specified as a hint (e.g. full framebuffer, half
     * framebuffer, etc.) or as an explicit dimension in pixels.
     * @param height The height of the output framebuffer. This can be specified as a hint (e.g. full framebuffer, half
     * framebuffer, etc.) or as an explicit dimension in pixels.
     * @param handles The handles of the outputs that should be used as framebuffer attachments. The size of this vector
     * determines the number of attachments. The format of the attachments is determined by the resource description of
     * the outputs.
     */
    void setOutputFramebufferData(const TextureSize& width,
                                  const TextureSize& height,
                                  std::initializer_list<uint32_t> handles)
    {
        framebuffer_data.valid   = true;
        framebuffer_data.width   = width;
        framebuffer_data.height  = height;
        framebuffer_data.handles = handles;
    }
};

}    // namespace atcg