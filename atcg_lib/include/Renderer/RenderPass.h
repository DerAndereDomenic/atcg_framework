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
     *
     * @param name The name of the render pass
     */
    RenderPass(std::string_view name = "RenderPass") : _name(name)
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

protected:
    RenderFunction _render_f;
    SetupFunction _setup_f;
    Dictionary _inputs;
    Dictionary _data;
    Dictionary _output;
    std::string _name;
};

}    // namespace atcg