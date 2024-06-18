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

TEST(ColorTest, MatchingXApproximation)
{
    const float x_bar[] = {0.001368, 0.002236, 0.004243, 0.007650, 0.014310, 0.023190, 0.043510, 0.077630, 0.134380,
                           0.214770, 0.283900, 0.328500, 0.348280, 0.348060, 0.336200, 0.318700, 0.290800, 0.251100,
                           0.195360, 0.142100, 0.095640, 0.057950, 0.032010, 0.014700, 0.004900, 0.002400, 0.009300,
                           0.029100, 0.063270, 0.109600, 0.165500, 0.225750, 0.290400, 0.359700, 0.433450, 0.512050,
                           0.594500, 0.678400, 0.762100, 0.842500, 0.916300, 0.978600, 1.026300, 1.056700, 1.062200,
                           1.045600, 1.002600, 0.938400, 0.854450, 0.751400, 0.642400, 0.541900, 0.447900, 0.360800,
                           0.283500, 0.218700, 0.164900, 0.121200, 0.087400, 0.063600, 0.046770, 0.032900, 0.022700,
                           0.015840, 0.011359, 0.008111, 0.005790, 0.004109, 0.002899, 0.002049, 0.001440, 0.001000,
                           0.000690, 0.000476, 0.000332, 0.000235, 0.000166, 0.000117, 0.000083, 0.000059, 0.000042};

    int len           = sizeof(x_bar) / sizeof(float);
    float total_error = 0.0f;
    for(int i = 0; i < len; ++i)
    {
        float x_bar_d = x_bar[i];
        float x_bar_a = atcg::Color::color_matching_x(380.0f + 5.0f * i);

        total_error += (x_bar_a - x_bar_d) * (x_bar_a - x_bar_d);
    }

    total_error /= float(len);

    EXPECT_NEAR(total_error, 0.0f, 1e-3f);
}

TEST(ColorTest, MatchingYApproximation)
{
    const float y_bar[] = {0.000039, 0.000064, 0.000120, 0.000217, 0.000396, 0.000640, 0.001210, 0.002180, 0.004000,
                           0.007300, 0.011600, 0.016840, 0.023000, 0.029800, 0.038000, 0.048000, 0.060000, 0.073900,
                           0.090980, 0.112600, 0.139020, 0.169300, 0.208020, 0.258600, 0.323000, 0.407300, 0.503000,
                           0.608200, 0.710000, 0.793200, 0.862000, 0.914850, 0.954000, 0.980300, 0.994950, 1.000000,
                           0.995000, 0.978600, 0.952000, 0.915400, 0.870000, 0.816300, 0.757000, 0.694900, 0.631000,
                           0.566800, 0.503000, 0.441200, 0.381000, 0.321000, 0.265000, 0.217000, 0.175000, 0.138200,
                           0.107000, 0.081600, 0.061000, 0.044580, 0.032000, 0.023200, 0.017000, 0.011920, 0.008210,
                           0.005723, 0.004102, 0.002929, 0.002091, 0.001484, 0.001047, 0.000740, 0.000520, 0.000361,
                           0.000249, 0.000172, 0.000120, 0.000085, 0.000060, 0.000042, 0.000030, 0.000021, 0.000015};

    int len           = sizeof(y_bar) / sizeof(float);
    float total_error = 0.0f;
    for(int i = 0; i < len; ++i)
    {
        float y_bar_d = y_bar[i];
        float y_bar_a = atcg::Color::color_matching_y(380.0f + 5.0f * i);

        total_error += (y_bar_a - y_bar_d) * (y_bar_a - y_bar_d);
    }

    total_error /= float(len);

    EXPECT_NEAR(total_error, 0.0f, 1e-3f);
}

TEST(ColorTest, MatchingZApproximation)
{
    const float z_bar[] = {0.006450, 0.010550, 0.020050, 0.036210, 0.067850, 0.110200, 0.207400, 0.371300, 0.645600,
                           1.039050, 1.385600, 1.622960, 1.747060, 1.782600, 1.772110, 1.744100, 1.669200, 1.528100,
                           1.287640, 1.041900, 0.812950, 0.616200, 0.465180, 0.353300, 0.272000, 0.212300, 0.158200,
                           0.111700, 0.078250, 0.057250, 0.042160, 0.029840, 0.020300, 0.013400, 0.008750, 0.005750,
                           0.003900, 0.002750, 0.002100, 0.001800, 0.001650, 0.001400, 0.001100, 0.001000, 0.000800,
                           0.000600, 0.000340, 0.000240, 0.000190, 0.000100, 0.000050, 0.000030, 0.000020, 0.000010,
                           0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                           0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                           0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000};

    int len           = sizeof(z_bar) / sizeof(float);
    float total_error = 0.0f;
    for(int i = 0; i < len; ++i)
    {
        float z_bar_d = z_bar[i];
        float z_bar_a = atcg::Color::color_matching_z(380.0f + 5.0f * i);

        total_error += (z_bar_a - z_bar_d) * (z_bar_a - z_bar_d);
    }

    total_error /= float(len);

    EXPECT_NEAR(total_error, 0.0f, 1e-3f);
}

TEST(ColorTest, colorMatchingNormalizationY)
{
    float step_size  = 0.01f;
    float lambda_min = 380.0f;
    float lambda_max = 780.0f;

    float integral_y = 0.0f;

    for(float lambda = lambda_min; lambda <= lambda_max; lambda += step_size)
    {
        integral_y += atcg::Color::color_matching_y(lambda);
    }

    integral_y *= step_size;

    EXPECT_NEAR(integral_y, atcg::Constants::Y_integral<float>(), 1e-5f);
}

TEST(ColorTest, D65sRGBNormalization)
{
    float step_size  = 0.01f;
    float lambda_min = 380.0f;
    float lambda_max = 780.0f;

    float integral_x = 0.0f;
    float integral_y = 0.0f;
    float integral_z = 0.0f;

    for(float lambda = lambda_min; lambda <= lambda_max; lambda += step_size)
    {
        integral_x += atcg::Color::color_matching_x(lambda) * atcg::Color::D65(lambda);
        integral_y += atcg::Color::color_matching_y(lambda) * atcg::Color::D65(lambda);
        integral_z += atcg::Color::color_matching_z(lambda) * atcg::Color::D65(lambda);
    }

    integral_x *= step_size;
    integral_y *= step_size;
    integral_z *= step_size;

    glm::vec3 color = atcg::Color::XYZ_to_sRGB(glm::vec3(integral_x, integral_y, integral_z) / integral_y);

    EXPECT_NEAR(color.x, 1.0f, 1e-2f);
    EXPECT_NEAR(color.y, 1.0f, 1e-2f);
    EXPECT_NEAR(color.z, 1.0f, 1e-2f);
}

TEST(ColorTest, Roundtrip)
{
    float step_size  = 0.01f;
    float lambda_min = 380.0f;
    float lambda_max = 780.0f;

    float integral_x = 0.0f;
    float integral_y = 0.0f;
    float integral_z = 0.0f;

    float illum_y = 0.0f;

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);    // distribution in range [1, 6]

    glm::vec3 lRGB(uniform(rng), uniform(rng), uniform(rng));

    for(float lambda = lambda_min; lambda <= lambda_max; lambda += step_size)
    {
        glm::vec3 spectrum_weight =
            glm::vec3(atcg::Color::Sr(lambda), atcg::Color::Sg(lambda), atcg::Color::Sb(lambda));
        float spectrum = glm::dot(spectrum_weight, lRGB);

        integral_x += spectrum * atcg::Color::D65(lambda) * atcg::Color::color_matching_x(lambda);
        integral_y += spectrum * atcg::Color::D65(lambda) * atcg::Color::color_matching_y(lambda);
        integral_z += spectrum * atcg::Color::D65(lambda) * atcg::Color::color_matching_z(lambda);

        illum_y += atcg::Color::D65(lambda) * atcg::Color::color_matching_y(lambda);
    }

    illum_y *= step_size;
    integral_x *= step_size / illum_y;
    integral_y *= step_size / illum_y;
    integral_z *= step_size / illum_y;

    glm::vec3 rec = atcg::Color::XYZ_to_lRGB(glm::vec3(integral_x, integral_y, integral_z));

    EXPECT_NEAR(rec.x, lRGB.x, 1e-1f);
    EXPECT_NEAR(rec.y, lRGB.y, 1e-1f);
    EXPECT_NEAR(rec.z, lRGB.z, 1e-1f);
}