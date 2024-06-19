#include <gtest/gtest.h>
#include <Math/Functions.h>
#include <random>

TEST(FunctionTest, mapFloat)
{
    float value  = -2.0f + 5.0f / 2.0f;
    float mapped = atcg::Math::map(value, -2.0f, 3.0f, -1.0f, 1.0f);

    EXPECT_NEAR(mapped, 0.0f, 1e-5f);
}

TEST(FunctionTest, mapDouble)
{
    double value  = -2.0 + 5.0 / 2.0;
    double mapped = atcg::Math::map(value, -2.0, 3.0, -1.0, 1.0);

    EXPECT_NEAR(mapped, 0.0f, 1e-5);
}

TEST(FunctionTest, mapVec2)
{
    glm::vec2 value(0.5f, 1.0f);

    glm::vec2 mapped =
        atcg::Math::map(value, glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 2.0f), glm::vec2(-1.0f), glm::vec2(1.0f));

    EXPECT_NEAR(mapped.x, 0.0f, 1e-5f);
    EXPECT_NEAR(mapped.y, 0.0f, 1e-5f);
}

TEST(FunctionTest, mapVec3)
{
    glm::vec3 value(0.5f, 1.0f, -1.5f);

    glm::vec3 mapped = atcg::Math::map(value,
                                       glm::vec3(0.0f, 0.0f, -3.0f),
                                       glm::vec3(1.0f, 2.0f, 0.0f),
                                       glm::vec3(-1.0f),
                                       glm::vec3(1.0f));

    EXPECT_NEAR(mapped.x, 0.0f, 1e-5f);
    EXPECT_NEAR(mapped.y, 0.0f, 1e-5f);
    EXPECT_NEAR(mapped.z, 0.0f, 1e-5f);
}

TEST(FunctionTest, mapVec4)
{
    glm::vec4 value(0.5f, 1.0f, -1.5f, 0.5f);

    glm::vec4 mapped = atcg::Math::map(value,
                                       glm::vec4(0.0f, 0.0f, -3.0f, -1.0f),
                                       glm::vec4(1.0f, 2.0f, 0.0f, 1.0f),
                                       glm::vec4(-1.0f),
                                       glm::vec4(1.0f));

    EXPECT_NEAR(mapped.x, 0.0f, 1e-5f);
    EXPECT_NEAR(mapped.y, 0.0f, 1e-5f);
    EXPECT_NEAR(mapped.z, 0.0f, 1e-5f);
    EXPECT_NEAR(mapped.w, 0.5f, 1e-5f);
}

TEST(FunctionTest, uv2ndcFloat)
{
    float value = 0.0f;

    float mapped = atcg::Math::uv2ndc(value);

    EXPECT_NEAR(mapped, -1.0f, 1e-5f);
}

TEST(FunctionTest, uv2ndcVec)
{
    glm::vec4 value(0.0f, 0.25f, 0.5f, 1.0f);

    glm::vec4 mapped = atcg::Math::uv2ndc(value);

    EXPECT_NEAR(mapped.x, -1.0f, 1e-5f);
    EXPECT_NEAR(mapped.y, -0.5f, 1e-5f);
    EXPECT_NEAR(mapped.z, 0.0f, 1e-5f);
    EXPECT_NEAR(mapped.w, 1.0f, 1e-5f);
}

TEST(FunctionTest, ndc2uvFloat)
{
    float value = 0.0f;

    float mapped = atcg::Math::ndc2uv(value);

    EXPECT_NEAR(mapped, 0.5f, 1e-5f);
}

TEST(FunctionTest, ndc2uvVec)
{
    glm::vec4 value(-1.0f, -0.5f, 0.0f, 0.5f);

    glm::vec4 mapped = atcg::Math::ndc2uv(value);

    EXPECT_NEAR(mapped.x, 0.0f, 1e-5f);
    EXPECT_NEAR(mapped.y, 0.25f, 1e-5f);
    EXPECT_NEAR(mapped.z, 0.5f, 1e-5f);
    EXPECT_NEAR(mapped.w, 0.75f, 1e-5f);
}

TEST(FunctionTest, ndcDepthRound)
{
    float n = 0.01f;
    float f = 1000.0f;

    float ndc_depth = 0.9997f;

    float linear_depth = atcg::Math::ndc2linearDepth(ndc_depth, n, f);
    float ndc_rec      = atcg::Math::linearDepth2ndc(linear_depth, n, f);

    EXPECT_NEAR(ndc_depth, ndc_rec, 1e-4f);
}

TEST(FunctionTest, linearDepthRound)
{
    float n = 0.01f;
    float f = 1000.0f;

    float linear_depth = 3.0f;

    float ndc_depth  = atcg::Math::linearDepth2ndc(linear_depth, n, f);
    float linear_rec = atcg::Math::ndc2linearDepth(ndc_depth, n, f);

    EXPECT_NEAR(linear_depth, linear_rec, 1e-4f);
}