#pragma once

#include <Core/Platform.h>
#include <Core/Memory.h>
#include <DataStructure/Dictionary.h>

#include <any>

namespace atcg
{

/**
 * @brief Base class for a Render Pass.
 * A render pass is a node in a DAG with sevaral inputs and one output. This base class is used to collect all different
 * render passes in a list. Use atcg::RenderPass to declare different intermediate- and output buffers.
 *
 */
class RenderPassBase
{
public:
    /**
     * @brief Setup the render pass
     *
     * @param context The render context
     */
    virtual void setup(Dictionary& context) = 0;

    /**
     * @brief Execute a reder pass
     *
     * @param context The render context
     */
    virtual void execute(Dictionary& context) = 0;

    /**
     * @brief Add an input to the Render pass.
     * This can be any type but if this node is not a root node, it will be an instance of atcg::ref_ptr
     *
     * @param input The input
     */
    virtual void addInput(std::any input) = 0;

    /**
     * @brief Get the output.
     * When creating a render pass, an atcg::ref_ptr will be allocated that holds the output
     *
     * @return An atcg::ref_ptr with type specified by the instance of atcg::RenderPass, i.e.,
     * atcg::ref_ptr<RenderPassOutputT>
     */
    virtual std::any getOutput() const = 0;
};

/**
 * @brief A class to model a render pass
 *
 * @tparam RenderPassOutputT The output type of the RenderPass
 */
template<typename RenderPassOutputT>
class RenderPass : public RenderPassBase
{
public:
    using RenderFunction = std::function<
        void(Dictionary&, const std::vector<std::any>&, Dictionary&, const atcg::ref_ptr<RenderPassOutputT>&)>;

    using SetupFunction = std::function<void(Dictionary&, Dictionary&, atcg::ref_ptr<RenderPassOutputT>&)>;

    /**
     * @brief Default constructor
     */
    RenderPass()
    {
        _output = atcg::make_ref<RenderPassOutputT>();

        _render_f = [](Dictionary&,
                       const std::vector<std::any>&,
                       Dictionary&,
                       const atcg::ref_ptr<RenderPassOutputT>&) {
        };

        _setup_f = [](Dictionary&,
                      Dictionary&,
                      atcg::ref_ptr<RenderPassOutputT>&) {
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
    ATCG_INLINE virtual void setup(Dictionary& context) override { _setup_f(context, _data, _output); }

    /**
     * @brief Execute a reder pass
     *
     * @param context The render context
     */
    ATCG_INLINE virtual void execute(Dictionary& context) override { _render_f(context, _inputs, _data, _output); }

    /**
     * @brief Add an input to the Render pass.
     * This can be any type but if this node is not a root node, it will be an instance of atcg::ref_ptr
     *
     * @param input The input
     */
    ATCG_INLINE virtual void addInput(std::any input) override { _inputs.push_back(input); }

    /**
     * @brief Get the output.
     * When creating a render pass, an atcg::ref_ptr will be allocated that holds the output
     *
     * @return An atcg::ref_ptr with type specified by the instance of atcg::RenderPass, i.e.,
     * atcg::ref_ptr<RenderPassOutputT>
     */
    ATCG_INLINE virtual std::any getOutput() const { return _output; }

private:
    RenderFunction _render_f;
    SetupFunction _setup_f;
    std::vector<std::any> _inputs;
    Dictionary _data;
    atcg::ref_ptr<RenderPassOutputT> _output;
};

/**
 * @brief A base class to build a render pass.
 * This class collects all the data for a render pass and then executes the setup code when calling build().
 * The type of the data is specified in the instance of atcg::RenderPassBuilder.
 *
 */
class RenderPassBuilderBase
{
public:
    /**
     * @brief Build the Render Pass
     *
     * @param context The Render context data
     * @return The compiled render pass
     */
    virtual atcg::ref_ptr<RenderPassBase> build(Dictionary& context) = 0;

    /**
     * @brief Add an input to the Render pass.
     * This can be any type but if this node is not a root node, it will be an instance of atcg::ref_ptr
     *
     * @param input The input
     */
    virtual void addInput(std::any input) = 0;

    /**
     * @brief Get the output.
     * When creating a render pass, an atcg::ref_ptr will be allocated that holds the output
     *
     * @return An atcg::ref_ptr with type specified by the instance of atcg::RenderPass, i.e.,
     * atcg::ref_ptr<RenderPassOutputT>
     */
    virtual std::any getOutput() const = 0;
};

/**
 * @brief A class to build a render pass.
 * This class collects all the data for a render pass and then executes the setup code when calling build().
 *
 * @tparam RenderPassOutputT The output type of the RenderPass
 */
template<typename RenderPassOutputT>
class RenderPassBuilder : public RenderPassBuilderBase
{
public:
    /**
     * @brief Default constructor
     */
    RenderPassBuilder() { _pass = atcg::make_ref<RenderPass<RenderPassOutputT>>(); }

    /**
     * @brief Set the render function.
     * This function gets called each time the RenderGraph is executed
     *
     * @param f The render function
     */
    ATCG_INLINE void setRenderFunction(
        std::function<
            void(Dictionary&, const std::vector<std::any>&, Dictionary&, const atcg::ref_ptr<RenderPassOutputT>&)> f)
    {
        _pass->setRenderFunction(f);
    }

    /**
     * @brief Set the setup function.
     * This function gets called when the RenderPass gets compiled
     *
     * @param f The setup function
     */
    ATCG_INLINE void
    setSetupFunction(std::function<void(Dictionary&, Dictionary&, atcg::ref_ptr<RenderPassOutputT>&)> f)
    {
        _pass->setSetupFunction(f);
    }

    /**
     * @brief Add an input to the Render pass.
     * This can be any type but if this node is not a root node, it will be an instance of atcg::ref_ptr
     *
     * @param input The input
     */
    ATCG_INLINE virtual void addInput(std::any input) override { _pass->addInput(input); }

    /**
     * @brief Get the output.
     * When creating a render pass, an atcg::ref_ptr will be allocated that holds the output
     *
     * @return An atcg::ref_ptr with type specified by the instance of atcg::RenderPass, i.e.,
     * atcg::ref_ptr<RenderPassOutputT>
     */
    ATCG_INLINE virtual std::any getOutput() const override { return _pass->getOutput(); }

    /**
     * @brief Build the Render Pass
     *
     * @param context The Render context data
     * @return The compiled render pass
     */
    ATCG_INLINE virtual atcg::ref_ptr<RenderPassBase> build(Dictionary& context) override
    {
        _pass->setup(context);
        return _pass;
    }

private:
    atcg::ref_ptr<RenderPass<RenderPassOutputT>> _pass;
};

}    // namespace atcg