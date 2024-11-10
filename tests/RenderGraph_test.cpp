#include <gtest/gtest.h>
#include <Renderer/RenderGraph.h>

TEST(RenderGraphTest, emptyGraph)
{
    atcg::ref_ptr<int> ctx = atcg::make_ref<int>();
    atcg::RenderGraph graph(ctx);

    graph.compile();
    graph.execute();
}

TEST(RenderGraphTest, singleEmptyNode)
{
    atcg::ref_ptr<int> ctx = atcg::make_ref<int>();
    atcg::RenderGraph graph(ctx);

    auto [handle, builder] = graph.addRenderPass<int, int>();

    graph.compile();
    graph.execute();
}

TEST(RenderGraphTest, singleNodeNoSetup)
{
    atcg::ref_ptr<int> ctx = atcg::make_ref<int>(5);
    atcg::RenderGraph graph(ctx);

    auto [handle, builder] = graph.addRenderPass<int, int>();

    builder->setRenderFunction([](const atcg::ref_ptr<int>& context,
                                  const std::vector<std::any>& inputs,
                                  const atcg::ref_ptr<int>& data,
                                  const atcg::ref_ptr<int>& output) { *output = 2 * (*context); });

    graph.compile();
    graph.execute();

    auto output = std::any_cast<atcg::ref_ptr<int>>(builder->getOutput());

    EXPECT_EQ(*output, 10);
}

TEST(RenderGraphTest, singleNodeSetup)
{
    atcg::ref_ptr<int> ctx = atcg::make_ref<int>();
    atcg::RenderGraph graph(ctx);

    auto [handle, builder] = graph.addRenderPass<int, int>();

    builder->setSetupFunction(
        [](const atcg::ref_ptr<int>& context, const atcg::ref_ptr<int>& data, atcg::ref_ptr<int>& output)
        { *data = 5; });

    builder->setRenderFunction([](const atcg::ref_ptr<int>& context,
                                  const std::vector<std::any>& inputs,
                                  const atcg::ref_ptr<int>& data,
                                  const atcg::ref_ptr<int>& output) { *output = 2 * (*data); });

    graph.compile();
    graph.execute();

    auto output = std::any_cast<atcg::ref_ptr<int>>(builder->getOutput());

    EXPECT_EQ(*output, 10);
}

TEST(RenderGraphTest, gaussGraph)
{
    struct RenderContext
    {
        float* x;
    };

    float x                          = 5.0f;
    atcg::ref_ptr<RenderContext> ctx = atcg::make_ref<RenderContext>();
    ctx->x                           = &x;

    // 1. Create render graph
    atcg::RenderGraph graph(ctx);

    auto [mu_handle, mu_builder]                 = graph.addRenderPass<int, float>();
    auto [sigma_handle, sigma_builder]           = graph.addRenderPass<int, float>();
    auto [normalize_handle, normalize_builder]   = graph.addRenderPass<int, float>();
    auto [two_square_handle, two_square_builder] = graph.addRenderPass<int, float>();
    auto [subtract_handle, subtract_builder]     = graph.addRenderPass<int, float>();
    auto [neg_square_handle, neg_square_builder] = graph.addRenderPass<int, float>();
    auto [div_handle, div_builder]               = graph.addRenderPass<int, float>();
    auto [exp_handle, exp_builder]               = graph.addRenderPass<int, float>();
    auto [mul_handle, mul_builder]               = graph.addRenderPass<int, float>();

    graph.addDependency(sigma_handle, normalize_handle);
    graph.addDependency(sigma_handle, two_square_handle);
    graph.addDependency(mu_handle, subtract_handle);
    graph.addDependency(subtract_handle, neg_square_handle);
    graph.addDependency(neg_square_handle, div_handle);
    graph.addDependency(two_square_handle, div_handle);
    graph.addDependency(div_handle, exp_handle);
    graph.addDependency(exp_handle, mul_handle);
    graph.addDependency(normalize_handle, mul_handle);

    mu_builder->addInput(1.0f);
    mu_builder->setRenderFunction([](const atcg::ref_ptr<RenderContext>& context,
                                     const std::vector<std::any>& inputs,
                                     const atcg::ref_ptr<int>&,
                                     const atcg::ref_ptr<float>& output)
                                  { *output = std::any_cast<float>(inputs[0]); });

    sigma_builder->addInput(3.0f);
    sigma_builder->setRenderFunction([](const atcg::ref_ptr<RenderContext>& context,
                                        const std::vector<std::any>& inputs,
                                        const atcg::ref_ptr<int>&,
                                        const atcg::ref_ptr<float>& output)
                                     { *output = std::any_cast<float>(inputs[0]); });

    normalize_builder->setRenderFunction(
        [](const atcg::ref_ptr<RenderContext>& context,
           const std::vector<std::any>& inputs,
           const atcg::ref_ptr<int>&,
           const atcg::ref_ptr<float>& output)
        {
            auto sigma = *std::any_cast<atcg::ref_ptr<float>>(inputs[0]);
            *output    = 1.0f / (std::sqrt(2.0f * 3.14159f) * sigma);
        });

    two_square_builder->setRenderFunction(
        [](const atcg::ref_ptr<RenderContext>& context,
           const std::vector<std::any>& inputs,
           const atcg::ref_ptr<int>&,
           const atcg::ref_ptr<float>& output)
        {
            auto sigma = *std::any_cast<atcg::ref_ptr<float>>(inputs[0]);
            *output    = 2.0f * sigma * sigma;
        });

    subtract_builder->setRenderFunction(
        [](const atcg::ref_ptr<RenderContext>& context,
           const std::vector<std::any>& inputs,
           const atcg::ref_ptr<int>&,
           const atcg::ref_ptr<float>& output)
        {
            auto x  = *(context->x);
            auto mu = *std::any_cast<atcg::ref_ptr<float>>(inputs[0]);
            *output = x - mu;
        });

    neg_square_builder->setRenderFunction(
        [](const atcg::ref_ptr<RenderContext>& context,
           const std::vector<std::any>& inputs,
           const atcg::ref_ptr<int>&,
           const atcg::ref_ptr<float>& output)
        {
            auto x  = *std::any_cast<atcg::ref_ptr<float>>(inputs[0]);
            *output = -x * x;
        });

    div_builder->setRenderFunction(
        [](const atcg::ref_ptr<RenderContext>& context,
           const std::vector<std::any>& inputs,
           const atcg::ref_ptr<int>&,
           const atcg::ref_ptr<float>& output)
        {
            auto x  = *std::any_cast<atcg::ref_ptr<float>>(inputs[0]);
            auto y  = *std::any_cast<atcg::ref_ptr<float>>(inputs[1]);
            *output = x / y;
        });

    exp_builder->setRenderFunction(
        [](const atcg::ref_ptr<RenderContext>& context,
           const std::vector<std::any>& inputs,
           const atcg::ref_ptr<int>&,
           const atcg::ref_ptr<float>& output)
        {
            auto x  = *std::any_cast<atcg::ref_ptr<float>>(inputs[0]);
            *output = std::exp(x);
        });

    mul_builder->setRenderFunction(
        [](const atcg::ref_ptr<RenderContext>& context,
           const std::vector<std::any>& inputs,
           const atcg::ref_ptr<int>&,
           const atcg::ref_ptr<float>& output)
        {
            auto x  = *std::any_cast<atcg::ref_ptr<float>>(inputs[0]);
            auto y  = *std::any_cast<atcg::ref_ptr<float>>(inputs[1]);
            *output = x * y;
        });

    graph.compile();

    auto gauss = [](const float x, const float mu, const float sigma) -> float
    {
        return 1.0f / (std::sqrt(2.0f * 3.14159f) * sigma) * std::exp(-(x - mu) * (x - mu) / (2.0f * sigma * sigma));
    };

    for(int i = 0; i < 10; ++i)
    {
        x = float(i);
        graph.execute();
        auto output = std::any_cast<atcg::ref_ptr<float>>(mul_builder->getOutput());
        float out   = *output;
        float exp   = gauss(x, 1.0f, 3.0f);
        EXPECT_NEAR(out, exp, 1e-5f);
    }
}