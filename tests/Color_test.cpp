#include <gtest/gtest.h>
#include <Math/Color.h>
#include <random>

TEST(ColorTest, quantizeWhite)
{
    glm::vec3 color(1.0f);
    glm::u8vec3 quantized = atcg::Color::quantize(color);

    EXPECT_EQ(quantized.x, 255);
    EXPECT_EQ(quantized.y, 255);
    EXPECT_EQ(quantized.z, 255);
}

TEST(ColorTest, quantizePurple)
{
    glm::vec3 color       = glm::vec3(108.0f, 18.0f, 150.0f) / 255.0f;
    glm::u8vec3 quantized = atcg::Color::quantize(color);

    EXPECT_EQ(quantized.x, 108);
    EXPECT_EQ(quantized.y, 18);
    EXPECT_EQ(quantized.z, 150);
}

TEST(ColorTest, dequantizeWhite)
{
    glm::u8vec3 color(255);
    glm::vec3 dequantized = atcg::Color::dequantize(color);

    EXPECT_EQ(dequantized.x, 1.0f);
    EXPECT_EQ(dequantized.y, 1.0f);
    EXPECT_EQ(dequantized.z, 1.0f);
}

TEST(ColorTest, dequantizePurple)
{
    glm::u8vec3 color     = glm::u8vec3(108, 18, 150);
    glm::vec3 dequantized = atcg::Color::dequantize(color);

    EXPECT_EQ(dequantized.x, 108.0f / 255.0f);
    EXPECT_EQ(dequantized.y, 18.0f / 255.0f);
    EXPECT_EQ(dequantized.z, 150.0f / 255.0f);
}

TEST(ColorTest, RGBConversion)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);    // distribution in range [1, 6]

    glm::vec3 lRGB(uniform(rng), uniform(rng), uniform(rng));

    glm::vec3 sRGB     = atcg::Color::lRGB_to_sRGB(lRGB);
    glm::vec3 lRGB_rec = atcg::Color::sRGB_to_lRGB(sRGB);

    EXPECT_NEAR(lRGB.x, lRGB_rec.x, 1e-3f);
    EXPECT_NEAR(lRGB.y, lRGB_rec.y, 1e-3f);
    EXPECT_NEAR(lRGB.z, lRGB_rec.z, 1e-3f);
}

TEST(ColorTest, lRGBXYZConversion)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);    // distribution in range [1, 6]

    glm::vec3 lRGB(uniform(rng), uniform(rng), uniform(rng));

    glm::vec3 XYZ      = atcg::Color::lRGB_to_XYZ(lRGB);
    glm::vec3 lRGB_rec = atcg::Color::XYZ_to_lRGB(XYZ);

    EXPECT_NEAR(lRGB.x, lRGB_rec.x, 1e-3f);
    EXPECT_NEAR(lRGB.y, lRGB_rec.y, 1e-3f);
    EXPECT_NEAR(lRGB.z, lRGB_rec.z, 1e-3f);
}

TEST(ColorTest, sRGBXYZConversion)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);    // distribution in range [1, 6]

    glm::vec3 sRGB(uniform(rng), uniform(rng), uniform(rng));

    glm::vec3 XYZ      = atcg::Color::sRGB_to_XYZ(sRGB);
    glm::vec3 sRGB_rec = atcg::Color::XYZ_to_sRGB(XYZ);

    EXPECT_NEAR(sRGB.x, sRGB_rec.x, 1e-3f);
    EXPECT_NEAR(sRGB.y, sRGB_rec.y, 1e-3f);
    EXPECT_NEAR(sRGB.z, sRGB_rec.z, 1e-3f);
}

TEST(ColorTest, D65Normalization)
{
    float response = atcg::Color::D65(560.0f);

    EXPECT_NEAR(response, 100.0f, 1.0f);
}