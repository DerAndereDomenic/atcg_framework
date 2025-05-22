#include <gtest/gtest.h>
#include <Renderer/RenderGraph.h>

TEST(RenderGraphTest, emptyGraph)
{
    atcg::RenderGraph graph;

    atcg::Dictionary context;
    graph.compile(context);
    graph.execute(context);
}

TEST(RenderGraphTest, singleEmptyNode)
{
    atcg::RenderGraph graph;

    auto [handle, builder] = graph.addRenderPass();

    atcg::Dictionary context;
    graph.compile(context);
    graph.execute(context);
}

TEST(RenderGraphTest, singleNodeNoSetup)
{
    atcg::RenderGraph graph;

    auto [handle, builder] = graph.addRenderPass();

    builder->setRenderFunction(
        [](atcg::Dictionary& context, const atcg::Dictionary& inputs, atcg::Dictionary& data, atcg::Dictionary& output)
        { output.setValue("output", 2 * context.getValue<int>("input")); });

    atcg::Dictionary context;
    context.setValue("input", 5);
    graph.compile(context);
    graph.execute(context);

    int output = builder->getOutputs().getValue<int>("output");

    EXPECT_EQ(output, 10);
}

TEST(RenderGraphTest, singleNodeSetup)
{
    atcg::RenderGraph graph;

    auto [handle, builder] = graph.addRenderPass();

    builder->setSetupFunction([](atcg::Dictionary& context, atcg::Dictionary& data, atcg::Dictionary& output)
                              { data.setValue("input", 5); });

    builder->setRenderFunction(
        [](atcg::Dictionary& context, const atcg::Dictionary& inputs, atcg::Dictionary& data, atcg::Dictionary& output)
        { output.setValue("output", 2 * data.getValue<int>("input")); });

    atcg::Dictionary context;
    graph.compile(context);
    graph.execute(context);

    int output = builder->getOutputs().getValue<int>("output");

    EXPECT_EQ(output, 10);
}

TEST(RenderGraphTest, gaussGraph)
{
    float x = 5.0f;
    atcg::Dictionary context;
    context.setValue("x", x);

    // 1. Create render graph
    atcg::RenderGraph graph;

    auto [mu_handle, mu_builder]                 = graph.addRenderPass("mu");
    auto [sigma_handle, sigma_builder]           = graph.addRenderPass("sigma");
    auto [normalize_handle, normalize_builder]   = graph.addRenderPass("normalize");
    auto [two_square_handle, two_square_builder] = graph.addRenderPass("two_square");
    auto [subtract_handle, subtract_builder]     = graph.addRenderPass("subtract");
    auto [neg_square_handle, neg_square_builder] = graph.addRenderPass("neg_square");
    auto [div_handle, div_builder]               = graph.addRenderPass("div");
    auto [exp_handle, exp_builder]               = graph.addRenderPass("exp");
    auto [mul_handle, mul_builder]               = graph.addRenderPass("mul");

    mu_builder->addInput("mu_input", 1.0f)
        ->setSetupFunction([](atcg::Dictionary& context, atcg::Dictionary& data, atcg::Dictionary& output)
                           { output.setValue("mu_output", atcg::make_ref<float>(0.0f)); })
        ->setRenderFunction(
            [](atcg::Dictionary& context,
               const atcg::Dictionary& inputs,
               atcg::Dictionary& data,
               atcg::Dictionary& outputs)
            {
                auto input  = inputs.getValue<float>("mu_input");
                auto output = outputs.getValue<atcg::ref_ptr<float>>("mu_output");

                *output = input;
            });

    sigma_builder->addInput("sigma_input", 3.0f)
        ->setSetupFunction([](atcg::Dictionary& context, atcg::Dictionary& data, atcg::Dictionary& output)
                           { output.setValue("sigma_output", atcg::make_ref<float>(0.0f)); })
        ->setRenderFunction(
            [](atcg::Dictionary& context,
               const atcg::Dictionary& inputs,
               atcg::Dictionary& data,
               atcg::Dictionary& outputs)
            {
                auto input  = inputs.getValue<float>("sigma_input");
                auto output = outputs.getValue<atcg::ref_ptr<float>>("sigma_output");

                *output = input;
            });

    normalize_builder
        ->setSetupFunction([](atcg::Dictionary& context, atcg::Dictionary& data, atcg::Dictionary& output)
                           { output.setValue("normalize_output", atcg::make_ref<float>(0.0f)); })
        ->setRenderFunction(
            [](atcg::Dictionary& context,
               const atcg::Dictionary& inputs,
               atcg::Dictionary& data,
               atcg::Dictionary& outputs)
            {
                auto output = outputs.getValue<atcg::ref_ptr<float>>("normalize_output");
                auto sigma  = *inputs.getValue<atcg::ref_ptr<float>>("sigma_input");
                *output     = 1.0f / (std::sqrt(2.0f * 3.14159f) * sigma);
            });

    two_square_builder
        ->setSetupFunction([](atcg::Dictionary& context, atcg::Dictionary& data, atcg::Dictionary& output)
                           { output.setValue("two_square_output", atcg::make_ref<float>(0.0f)); })
        ->setRenderFunction(
            [](atcg::Dictionary& context,
               const atcg::Dictionary& inputs,
               atcg::Dictionary& data,
               atcg::Dictionary& outputs)
            {
                auto output = outputs.getValue<atcg::ref_ptr<float>>("two_square_output");
                auto x      = *inputs.getValue<atcg::ref_ptr<float>>("x");
                *output     = 2.0f * x * x;
            });

    subtract_builder
        ->setSetupFunction([](atcg::Dictionary& context, atcg::Dictionary& data, atcg::Dictionary& output)
                           { output.setValue("subtract_output", atcg::make_ref<float>(0.0f)); })
        ->setRenderFunction(
            [](atcg::Dictionary& context,
               const atcg::Dictionary& inputs,
               atcg::Dictionary& data,
               atcg::Dictionary& outputs)
            {
                auto output = outputs.getValue<atcg::ref_ptr<float>>("subtract_output");
                auto x      = context.getValue<float>("x");
                auto mu     = *inputs.getValue<atcg::ref_ptr<float>>("mu_input");
                *output     = x - mu;
            });

    neg_square_builder
        ->setSetupFunction([](atcg::Dictionary& context, atcg::Dictionary& data, atcg::Dictionary& output)
                           { output.setValue("neg_square_output", atcg::make_ref<float>(0.0f)); })
        ->setRenderFunction(
            [](atcg::Dictionary& context,
               const atcg::Dictionary& inputs,
               atcg::Dictionary& data,
               atcg::Dictionary& outputs)
            {
                auto output = outputs.getValue<atcg::ref_ptr<float>>("neg_square_output");
                auto x      = *inputs.getValue<atcg::ref_ptr<float>>("x");
                *output     = -x * x;
            });

    div_builder
        ->setSetupFunction([](atcg::Dictionary& context, atcg::Dictionary& data, atcg::Dictionary& output)
                           { output.setValue("div_output", atcg::make_ref<float>(0.0f)); })
        ->setRenderFunction(
            [](atcg::Dictionary& context,
               const atcg::Dictionary& inputs,
               atcg::Dictionary& data,
               atcg::Dictionary& outputs)
            {
                auto output = outputs.getValue<atcg::ref_ptr<float>>("div_output");
                auto x      = *inputs.getValue<atcg::ref_ptr<float>>("x");
                auto y      = *inputs.getValue<atcg::ref_ptr<float>>("y");
                *output     = x / y;
            });

    exp_builder
        ->setSetupFunction([](atcg::Dictionary& context, atcg::Dictionary& data, atcg::Dictionary& output)
                           { output.setValue("exp_output", atcg::make_ref<float>(0.0f)); })
        ->setRenderFunction(
            [](atcg::Dictionary& context,
               const atcg::Dictionary& inputs,
               atcg::Dictionary& data,
               atcg::Dictionary& outputs)
            {
                auto output = outputs.getValue<atcg::ref_ptr<float>>("exp_output");
                auto x      = *inputs.getValue<atcg::ref_ptr<float>>("x");
                *output     = std::exp(x);
            });

    mul_builder
        ->setSetupFunction([](atcg::Dictionary& context, atcg::Dictionary& data, atcg::Dictionary& output)
                           { output.setValue("mul_output", atcg::make_ref<float>(0.0f)); })
        ->setRenderFunction(
            [](atcg::Dictionary& context,
               const atcg::Dictionary& inputs,
               atcg::Dictionary& data,
               atcg::Dictionary& outputs)
            {
                auto output = outputs.getValue<atcg::ref_ptr<float>>("mul_output");
                auto x      = *inputs.getValue<atcg::ref_ptr<float>>("x");
                auto y      = *inputs.getValue<atcg::ref_ptr<float>>("y");
                *output     = x * y;
            });

    graph.addDependency(sigma_handle, "sigma_output", normalize_handle, "sigma_input");
    graph.addDependency(sigma_handle, "sigma_output", two_square_handle, "x");
    graph.addDependency(mu_handle, "mu_output", subtract_handle, "mu_input");
    graph.addDependency(subtract_handle, "subtract_output", neg_square_handle, "x");
    graph.addDependency(neg_square_handle, "neg_square_output", div_handle, "x");
    graph.addDependency(two_square_handle, "two_square_output", div_handle, "y");
    graph.addDependency(div_handle, "div_output", exp_handle, "x");
    graph.addDependency(exp_handle, "exp_output", mul_handle, "x");
    graph.addDependency(normalize_handle, "normalize_output", mul_handle, "y");

    graph.compile(context);
    graph.exportToDOT("Test.dot");

    auto gauss = [](const float x, const float mu, const float sigma) -> float
    {
        return 1.0f / (std::sqrt(2.0f * 3.14159f) * sigma) * std::exp(-(x - mu) * (x - mu) / (2.0f * sigma * sigma));
    };

    for(int i = 0; i < 10; ++i)
    {
        context.setValue("x", float(i));
        graph.execute(context);
        auto output = mul_builder->getOutputs().getValue<atcg::ref_ptr<float>>("mul_output");
        float out   = *output;
        float exp   = gauss(float(i), 1.0f, 3.0f);
        EXPECT_NEAR(out, exp, 1e-5f);
    }
}