#pragma once

#include <Core/Platform.h>
#include <Core/Memory.h>
#include <DataStructure/Dictionary.h>
#include <Renderer/Framebuffer.h>
#include <DataStructure/ResourcePool.h>

#include <any>

namespace atcg
{

enum class RenderTargetMode
{
    RENDER_TARGET_BOUND_FRAMEBUFFER,
    RENDER_TARGET_INPUT_FRAMEBUFFER,
    RENDER_TARGET_OWN_FRAMEBUFFER
};

struct RenderTargetDesc
{
    RenderTargetDesc() = default;

    RenderTargetDesc(RenderTargetMode mode) : mode(mode) {}

    RenderTargetDesc(RenderTargetMode mode, bool clear) : mode(mode), clear(clear) {}

    RenderTargetDesc(RenderTargetMode mode, const FramebufferSpecification& spec) : mode(mode), target_spec(spec) {}

    RenderTargetDesc(RenderTargetMode mode, bool clear, const FramebufferSpecification& spec)
        : mode(mode),
          target_spec(spec),
          clear(clear)
    {
    }

    RenderTargetMode mode = RenderTargetMode::RENDER_TARGET_BOUND_FRAMEBUFFER;

    FramebufferSpecification target_spec = {};

    bool clear = false;
};

/**
 * @brief A class to model a render pass
 *
 * The output type of the RenderPass
 */
class RenderPass
{
public:
    // void render(Dictionary& context, const Dictionary& inputs, Dictionary& pass_data, Dictionary& output);
    using RenderFunction = std::function<void(Dictionary&, const Dictionary&, Dictionary&, Dictionary&)>;

    // void setup(Dictionary& context, Dictionary& pass_data, Dictionary& output);
    using SetupFunction = std::function<void(Dictionary&, Dictionary&, Dictionary&)>;

    /**
     * @brief Default constructor
     *
     * @param desc The render target description
     * @param name The name of the render pass
     */
    RenderPass(const RenderTargetDesc& desc, std::string_view name = "RenderPass") : _name(name), _render_target(desc)
    {
        _render_f = [](Dictionary&, const Dictionary&, Dictionary&, Dictionary&) {
        };

        _setup_f = [](Dictionary&, Dictionary&, Dictionary&) {
        };
    }

    /**
     * @brief Destructor
     */
    virtual ~RenderPass() = default;

    /**
     * @brief Set the setup function.
     * This function is called when the RenderGraph is compiled
     *
     * @param f The setup function
     * @return this
     */
    ATCG_INLINE RenderPass* setSetupFunction(SetupFunction f)
    {
        _setup_f = f;
        return this;
    }

    /**
     * @brief Set the render function.
     * This function is called when the RenderGraph is executed
     *
     * @param f The render function
     * @return this
     */
    ATCG_INLINE RenderPass* setRenderFunction(RenderFunction f)
    {
        _render_f = f;
        return this;
    }

    /**
     * @brief Add an input to the Render pass.
     *
     * @param port_name The input port name
     * @param input The input
     * @return this
     */
    ATCG_INLINE virtual RenderPass* addInput(std::string_view port_name, std::any input)
    {
        _inputs.setValue(port_name, input);
        return this;
    }

    /**
     * @brief Register an output variable
     *
     * @param port_name The output name
     * @param output The output variable
     * @return this
     */
    ATCG_INLINE virtual RenderPass* registerOutput(std::string_view port_name, std::any output)
    {
        _output.setValue(port_name, output);
        return this;
    }

    /**
     * @brief Setup the render pass
     *
     * @param context The render context
     */
    ATCG_INLINE virtual void setup(Dictionary& context) { _setup_f(context, _data, _output); }

    /**
     * @brief Execute a reder pass
     *
     * @param context The render context
     */
    ATCG_INLINE virtual void execute(Dictionary& context) { _render_f(context, _inputs, _data, _output); }

    /**
     * @brief Get the outputs.
     *
     * @return Dictionary containing the output variables to the corresponding ports
     */
    ATCG_INLINE virtual const Dictionary& getOutputs() const { return _output; }

    /**
     * @brief Get the name of this render pass
     *
     * @return The name
     */
    ATCG_INLINE const std::string& name() const { return _name; }

    /**
     * @brief Garbage collect.
     * This function increases the lifetime of garbage collected objects and destroys them if the maximum life time is
     * reached.
     */
    ATCG_INLINE void garbageCollect() { _pool.garbageCollect(); }

    ATCG_INLINE atcg::ref_ptr<Framebuffer>
    prepareFramebuffer(Dictionary& context, const Dictionary& inputs, Dictionary& data, Dictionary& outputs)
    {
        atcg::ref_ptr<Framebuffer> target_fb = nullptr;
        switch(_render_target.mode)
        {
            case RenderTargetMode::RENDER_TARGET_BOUND_FRAMEBUFFER:
            {
                target_fb = context.getValue<atcg::ref_ptr<Framebuffer>>("target");
                target_fb->use();
            }
            break;
            case RenderTargetMode::RENDER_TARGET_INPUT_FRAMEBUFFER:
            {
                auto target = inputs.getValue<atcg::ref_ptr<atcg::ref_ptr<Framebuffer>>>("framebuffer");
                (*target)->use();
                target_fb = *target;
            }
            break;
            case RenderTargetMode::RENDER_TARGET_OWN_FRAMEBUFFER:
            {
                auto target     = data.getValue<atcg::ref_ptr<atcg::ref_ptr<Framebuffer>>>("target");
                auto screen_fbo = context.getValue<atcg::ref_ptr<Framebuffer>>("target");

                _render_target.target_spec.width  = screen_fbo->width();
                _render_target.target_spec.height = screen_fbo->height();
                //*target                           = Framebuffer::create(_render_target.target_spec);    // TODO
                *target = _pool.acquireFramebuffer({"target", _render_target.target_spec});

                (*target)->use();
                target_fb = *target;
            }
            break;
        }

        return target_fb;
    }

protected:
    RenderFunction _render_f;
    SetupFunction _setup_f;
    Dictionary _inputs;
    Dictionary _data;
    Dictionary _output;
    std::string _name;

    ResourcePool _pool;
    RenderTargetDesc _render_target;
};

}    // namespace atcg