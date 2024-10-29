#pragma once

#include <Core/Platform.h>
#include <Core/Memory.h>

#include <any>

namespace atcg
{

/**
 * @brief Base class for a Render Pass.
 * A render pass is a node in a DAG with sevaral inputs and one output. This base class is to collect all different
 * render passes. Use atcg::RenderPass to declare different intermediate and output buffers.
 *
 * @tparam RenderContextT Context of the Renderer
 */
template<typename RenderContextT>
struct RenderPassBase
{
public:
    /**
     * @brief Setup the render pass
     *
     * @param context The render context
     */
    virtual void setup(const atcg::ref_ptr<RenderContextT>& context) = 0;

    /**
     * @brief Execute a reder pass
     *
     * @param context The render context
     */
    virtual void execute(const atcg::ref_ptr<RenderContextT>& context) = 0;

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
 * @tparam RenderContextT Context that holds Renderer data
 * @tparam RenderPassDataT A buffer that is local to each render pass
 * @tparam RenderPassOutputT The output type of the RenderPass
 */
template<typename RenderContextT, typename RenderPassDataT, typename RenderPassOutputT>
class RenderPass : public RenderPassBase<RenderContextT>
{
public:
    using RenderFunction = std::function<void(const atcg::ref_ptr<RenderContextT>&,
                                              const std::vector<std::any>&,
                                              const atcg::ref_ptr<RenderPassDataT>&,
                                              const atcg::ref_ptr<RenderPassOutputT>&)>;

    using SetupFunction = std::function<void(const atcg::ref_ptr<RenderContextT>&,
                                             const atcg::ref_ptr<RenderPassDataT>&,
                                             atcg::ref_ptr<RenderPassOutputT>&)>;

    /**
     * @brief Default constructor
     */
    RenderPass()
    {
        _output = atcg::make_ref<RenderPassOutputT>();
        _data   = atcg::make_ref<RenderPassDataT>();

        _render_f = [](const atcg::ref_ptr<RenderContextT>&,
                       const std::vector<std::any>&,
                       const atcg::ref_ptr<RenderPassDataT>&,
                       const atcg::ref_ptr<RenderPassOutputT>&) {
        };

        _setup_f = [](const atcg::ref_ptr<RenderContextT>&,
                      const atcg::ref_ptr<RenderPassDataT>&,
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
    ATCG_INLINE virtual void setup(const atcg::ref_ptr<RenderContextT>& context) override
    {
        _setup_f(context, _data, _output);
    }

    /**
     * @brief Execute a reder pass
     *
     * @param context The render context
     */
    ATCG_INLINE virtual void execute(const atcg::ref_ptr<RenderContextT>& context) override
    {
        _render_f(context, _inputs, _data, _output);
    }

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
    atcg::ref_ptr<RenderPassDataT> _data;
    atcg::ref_ptr<RenderPassOutputT> _output;
};

/**
 * @brief A base class to build a render pass.
 * This class collects all the data for a render pass and then executes the setup code when calling build().
 * The type of the data is specified in the instance of atcg::RenderPassBuilder.
 *
 * @tparam RenderContextT The Render context data
 */
template<typename RenderContextT>
class RenderPassBuilderBase
{
public:
    /**
     * @brief Build the Render Pass
     *
     * @param context The Render context data
     * @return The compiled render pass
     */
    virtual atcg::ref_ptr<RenderPassBase<RenderContextT>> build(const atcg::ref_ptr<RenderContextT>& context) = 0;

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
 * @tparam RenderContextT The Render context data
 * @tparam RenderPassDataT A buffer that is local to each render pass
 * @tparam RenderPassOutputT The output type of the RenderPass
 */
template<typename RenderContextT, typename RenderPassDataT, typename RenderPassOutputT>
class RenderPassBuilder : public RenderPassBuilderBase<RenderContextT>
{
public:
    /**
     * @brief Default constructor
     */
    RenderPassBuilder() { _pass = atcg::make_ref<RenderPass<RenderContextT, RenderPassDataT, RenderPassOutputT>>(); }

    /**
     * @brief Set the render function.
     * This function gets called each time the RenderGraph is executed
     *
     * @param f The render function
     */
    ATCG_INLINE void setRenderFunction(std::function<void(const atcg::ref_ptr<RenderContextT>&,
                                                          const std::vector<std::any>&,
                                                          const atcg::ref_ptr<RenderPassDataT>&,
                                                          const atcg::ref_ptr<RenderPassOutputT>&)> f)
    {
        _pass->setRenderFunction(f);
    }

    /**
     * @brief Set the setup function.
     * This function gets called when the RenderPass gets compiled
     *
     * @param f The setup function
     */
    ATCG_INLINE void setSetupFunction(std::function<void(const atcg::ref_ptr<RenderContextT>&,
                                                         const atcg::ref_ptr<RenderPassDataT>&,
                                                         atcg::ref_ptr<RenderPassOutputT>&)> f)
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
    ATCG_INLINE virtual atcg::ref_ptr<RenderPassBase<RenderContextT>>
    build(const atcg::ref_ptr<RenderContextT>& context) override
    {
        _pass->setup(context);
        return _pass;
    }

private:
    atcg::ref_ptr<RenderPass<RenderContextT, RenderPassDataT, RenderPassOutputT>> _pass;
};

}    // namespace atcg