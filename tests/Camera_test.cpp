#include <gtest/gtest.h>
#include <Renderer/CameraUtils.h>
#include <Core/Log.h>

TEST(CameraTest, conversionToOpencvBasic)
{
    uint32_t width  = 1920;
    uint32_t height = 1080;
    atcg::CameraIntrinsics intrinsics_before;
    intrinsics_before.setAspectRatio((float)width / ((float)height));

    glm::mat3 K = atcg::CameraUtils::convert_to_opencv(intrinsics_before, width, height);
    atcg::CameraIntrinsics intrinsics_after =
        atcg::CameraUtils::convert_from_opencv(K, intrinsics_before.zNear(), intrinsics_before.zFar(), width, height);

    EXPECT_NEAR(intrinsics_before.aspectRatio(), intrinsics_after.aspectRatio(), 1e-3f);
    EXPECT_NEAR(intrinsics_before.FOV(), intrinsics_after.FOV(), 1e-3f);
    EXPECT_NEAR(intrinsics_before.opticalCenter().x, intrinsics_after.opticalCenter().x, 1e-3f);
    EXPECT_NEAR(intrinsics_before.opticalCenter().y, intrinsics_after.opticalCenter().y, 1e-3f);
    EXPECT_NEAR(intrinsics_before.zFar(), intrinsics_after.zFar(), 10.0f);    // Numerical issues?
    EXPECT_NEAR(intrinsics_before.zNear(), intrinsics_after.zNear(), 1e-3f);
    for(int i = 0; i < 4; ++i)
    {
        for(int j = 0; j < 4; ++j)
        {
            EXPECT_NEAR(intrinsics_before.projection()[i][j], intrinsics_after.projection()[i][j], 1e-3f);
        }
    }
}

TEST(CameraTest, conversionToOpencvComplex)
{
    uint32_t width  = 1920;
    uint32_t height = 1080;
    atcg::CameraIntrinsics intrinsics_before;
    intrinsics_before.setAspectRatio((float)width / ((float)height));
    intrinsics_before.setFOV(48.0f);
    intrinsics_before.setOpticalCenter(glm::vec2(0.1f, -0.3f));

    glm::mat3 K = atcg::CameraUtils::convert_to_opencv(intrinsics_before, width, height);
    atcg::CameraIntrinsics intrinsics_after =
        atcg::CameraUtils::convert_from_opencv(K, intrinsics_before.zNear(), intrinsics_before.zFar(), width, height);

    EXPECT_NEAR(intrinsics_before.aspectRatio(), intrinsics_after.aspectRatio(), 1e-3f);
    EXPECT_NEAR(intrinsics_before.FOV(), intrinsics_after.FOV(), 1e-3f);
    EXPECT_NEAR(intrinsics_before.opticalCenter().x, intrinsics_after.opticalCenter().x, 1e-3f);
    EXPECT_NEAR(intrinsics_before.opticalCenter().y, intrinsics_after.opticalCenter().y, 1e-3f);
    EXPECT_NEAR(intrinsics_before.zFar(), intrinsics_after.zFar(), 10.0f);    // Numerical issues?
    EXPECT_NEAR(intrinsics_before.zNear(), intrinsics_after.zNear(), 1e-3f);
    for(int i = 0; i < 4; ++i)
    {
        for(int j = 0; j < 4; ++j)
        {
            EXPECT_NEAR(intrinsics_before.projection()[i][j], intrinsics_after.projection()[i][j], 1e-3f);
        }
    }
}

TEST(CameraTest, conversionToOpencvAspect)
{
    uint32_t width  = 1920;
    uint32_t height = 1080;
    atcg::CameraIntrinsics intrinsics_before;
    intrinsics_before.setAspectRatio(1.0f);

    glm::mat3 K = atcg::CameraUtils::convert_to_opencv(intrinsics_before, width, height);
    atcg::CameraIntrinsics intrinsics_after =
        atcg::CameraUtils::convert_from_opencv(K, intrinsics_before.zNear(), intrinsics_before.zFar(), width, height);

    EXPECT_NEAR(intrinsics_before.aspectRatio(), intrinsics_after.aspectRatio(), 1e-3f);
    EXPECT_NEAR(intrinsics_before.FOV(), intrinsics_after.FOV(), 1e-3f);
    EXPECT_NEAR(intrinsics_before.opticalCenter().x, intrinsics_after.opticalCenter().x, 1e-3f);
    EXPECT_NEAR(intrinsics_before.opticalCenter().y, intrinsics_after.opticalCenter().y, 1e-3f);
    EXPECT_NEAR(intrinsics_before.zFar(), intrinsics_after.zFar(), 10.0f);    // Numerical issues?
    EXPECT_NEAR(intrinsics_before.zNear(), intrinsics_after.zNear(), 1e-3f);
    for(int i = 0; i < 4; ++i)
    {
        for(int j = 0; j < 4; ++j)
        {
            EXPECT_NEAR(intrinsics_before.projection()[i][j], intrinsics_after.projection()[i][j], 1e-3f);
        }
    }
}

TEST(CameraTest, conversionFromOpencvBasic)
{
    uint32_t width  = 1920;
    uint32_t height = 1080;

    float fx = 800;
    float fy = 800;
    float cx = 960;
    float cy = 540;

    glm::mat3 K_before = glm::mat4(1);
    K_before[0][0]     = fx;
    K_before[1][1]     = fy;
    K_before[0][2]     = cx;
    K_before[1][2]     = cy;

    atcg::CameraIntrinsics intrinsics = atcg::CameraUtils::convert_from_opencv(K_before, 0.01f, 1000.0f, width, height);
    glm::mat3 K_after                 = atcg::CameraUtils::convert_to_opencv(intrinsics, width, height);

    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(K_before[i][j], K_after[i][j], 1e-3f);
        }
    }
}

TEST(CameraTest, conversionFromOpencvComplex)
{
    uint32_t width  = 1920;
    uint32_t height = 1080;

    float fx = 800;
    float fy = 750;
    float cx = 950;
    float cy = 590;

    glm::mat3 K_before = glm::mat4(1);
    K_before[0][0]     = fx;
    K_before[1][1]     = fy;
    K_before[0][2]     = cx;
    K_before[1][2]     = cy;

    atcg::CameraIntrinsics intrinsics = atcg::CameraUtils::convert_from_opencv(K_before, 0.01f, 1000.0f, width, height);
    glm::mat3 K_after                 = atcg::CameraUtils::convert_to_opencv(intrinsics, width, height);

    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(K_before[i][j], K_after[i][j], 1e-3f);
        }
    }
}