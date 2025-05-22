#pragma once

#include <Core/Platform.h>
#include <Core/Memory.h>
#include <DataStructure/Dictionary.h>

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
    // void render(Dictionary& context, const Dictionary& inputs, Dictionary& pass_data, Dictionary& output);
    using RenderFunction = std::function<void(Dictionary&, const Dictionary&, Dictionary&, Dictionary&)>;

    // void setup(Dictionary& context, Dictionary& pass_data, Dictionary& output);
    using SetupFunction = std::function<void(Dictionary&, Dictionary&, Dictionary&)>;

    /**
     * @brief Default constructor
     */
    RenderPass(std::string_view name = "RenderPass") : _name(name)
    {
        _render_f = [](Dictionary&, const Dictionary&, Dictionary&, Dictionary&) {
        };

        _setup_f = [](Dictionary&, Dictionary&, Dictionary&) {
        };
    }

    /**
     * @brief Set the setup function.
     * This function is called when the RenderGraph is compiled
     *
     * @param f The setup function
     */
    ATCG_INLINE void setSetupFunction(SetupFunction f) { _setup_f = f; }

    /**
     * @brief Set the render function.
     * This function is called when the RenderGraph is executed
     *
     * @param f The render function
     */
    ATCG_INLINE void setRenderFunction(RenderFunction f) { _render_f = f; }

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
     * @brief Add an input to the Render pass.
     * This can be any type but if this node is not a root node, it will be an instance of atcg::ref_ptr
     *
     * @param input The input
     */
    ATCG_INLINE virtual void addInput(std::string_view port_name, std::any input)
    {
        _inputs.setValue(port_name, input);
    }

    /**
     * @brief Get the output.
     * When creating a render pass, an atcg::ref_ptr will be allocated that holds the output
     *
     * @return An atcg::ref_ptr with type specified by the instance of atcg::RenderPass, i.e.,
     * atcg::ref_ptr<RenderPassOutputT>
     */
    ATCG_INLINE virtual const Dictionary& getOutputs() const { return _output; }

    /**
     * @brief Get the name of this render pass
     *
     * @return The name
     */
    ATCG_INLINE const std::string& name() const { return _name; }

private:
    RenderFunction _render_f;
    SetupFunction _setup_f;
    Dictionary _inputs;
    Dictionary _data;
    Dictionary _output;
    std::string _name;
};

/**
 * @brief A class to build a render pass.
 * This class collects all the data for a render pass and then executes the setup code when calling build().
 */
class RenderPassBuilder
{
public:
    /**
     * @brief Default constructor
     */
    RenderPassBuilder(std::string_view name = "") { _pass = atcg::make_ref<RenderPass>(name); }

    /**
     * @brief Set the render function.
     * This function gets called each time the RenderGraph is executed
     *
     * @param f The render function
     */
    ATCG_INLINE RenderPassBuilder* setRenderFunction(RenderPass::RenderFunction f)
    {
        _pass->setRenderFunction(f);
        return this;
    }

    /**
     * @brief Set the setup function.
     * This function gets called when the RenderPass gets compiled
     *
     * @param f The setup function
     */
    ATCG_INLINE RenderPassBuilder* setSetupFunction(RenderPass::SetupFunction f)
    {
        _pass->setSetupFunction(f);
        return this;
    }

    /**
     * @brief Add an input to the Render pass.
     * This can be any type but if this node is not a root node, it will be an instance of atcg::ref_ptr
     *
     * @param input The input
     */
    ATCG_INLINE virtual RenderPassBuilder* addInput(std::string_view port_name, std::any input)
    {
        _pass->addInput(port_name, input);
        return this;
    }

    /**
     * @brief Get the output.
     * When creating a render pass, an atcg::ref_ptr will be allocated that holds the output
     *
     * @return An atcg::ref_ptr with type specified by the instance of atcg::RenderPass, i.e.,
     * atcg::ref_ptr<RenderPassOutputT>
     */
    ATCG_INLINE virtual const Dictionary& getOutputs() const { return _pass->getOutputs(); }

    /**
     * @brief Build the Render Pass
     *
     * @param context The Render context data
     * @return The compiled render pass
     */
    ATCG_INLINE virtual atcg::ref_ptr<RenderPass> build(Dictionary& context)
    {
        _pass->setup(context);
        return _pass;
    }

    ATCG_INLINE const std::string& name() const {return _pass->name();}

private:
    atcg::ref_ptr<RenderPass> _pass;
};

}    // namespace atcg