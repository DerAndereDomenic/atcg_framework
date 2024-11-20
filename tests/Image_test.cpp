#include <gtest/gtest.h>
#include <Math/Functions.h>
#include <DataStructure/Image.h>
#include <DataStructure/TorchUtils.h>
#include <Core/Path.h>

TEST(ImageTest, rawBytes)
{
    uint8_t dummy[2 * 4 * 3];
    atcg::Image img(dummy, 2, 4, 3);

    EXPECT_EQ(img.width(), 2);
    EXPECT_EQ(img.height(), 4);
    EXPECT_EQ(img.channels(), 3);
    EXPECT_FALSE(img.isHDR());

    auto data_tensor = img.data();

    EXPECT_EQ(data_tensor.device().type(), atcg::CPU);
    EXPECT_EQ(data_tensor.scalar_type(), torch::kUInt8);
    EXPECT_EQ(data_tensor.size(0), 4);
    EXPECT_EQ(data_tensor.size(1), 2);
    EXPECT_EQ(data_tensor.size(2), 3);
}

TEST(ImageTest, rawFloat)
{
    float dummy[2 * 4 * 3];
    atcg::Image img(dummy, 2, 4, 3);

    EXPECT_EQ(img.width(), 2);
    EXPECT_EQ(img.height(), 4);
    EXPECT_EQ(img.channels(), 3);
    EXPECT_TRUE(img.isHDR());

    auto data_tensor = img.data();

    EXPECT_EQ(data_tensor.device().type(), atcg::CPU);
    EXPECT_EQ(data_tensor.scalar_type(), torch::kFloat32);
    EXPECT_EQ(data_tensor.size(0), 4);
    EXPECT_EQ(data_tensor.size(1), 2);
    EXPECT_EQ(data_tensor.size(2), 3);
}

TEST(ImageTest, tensorUint8)
{
    torch::Tensor dummy = torch::zeros({4, 2, 3}, atcg::TensorOptions::uint8HostOptions());
    atcg::Image img(dummy);

    EXPECT_EQ(img.width(), 2);
    EXPECT_EQ(img.height(), 4);
    EXPECT_EQ(img.channels(), 3);
    EXPECT_FALSE(img.isHDR());

    auto data_tensor = img.data();

    EXPECT_EQ(data_tensor.device().type(), atcg::CPU);
    EXPECT_EQ(data_tensor.scalar_type(), torch::kUInt8);
    EXPECT_EQ(data_tensor.size(0), 4);
    EXPECT_EQ(data_tensor.size(1), 2);
    EXPECT_EQ(data_tensor.size(2), 3);
}

TEST(ImageTest, tensorFloat)
{
    torch::Tensor dummy = torch::zeros({4, 2, 3}, atcg::TensorOptions::floatHostOptions());
    atcg::Image img(dummy);

    EXPECT_EQ(img.width(), 2);
    EXPECT_EQ(img.height(), 4);
    EXPECT_EQ(img.channels(), 3);
    EXPECT_TRUE(img.isHDR());

    auto data_tensor = img.data();

    EXPECT_EQ(data_tensor.device().type(), atcg::CPU);
    EXPECT_EQ(data_tensor.scalar_type(), torch::kFloat32);
    EXPECT_EQ(data_tensor.size(0), 4);
    EXPECT_EQ(data_tensor.size(1), 2);
    EXPECT_EQ(data_tensor.size(2), 3);
}

TEST(ImageTest, rawBytessetData)
{
    uint8_t dummy[2 * 4 * 3];
    atcg::Image img;
    img.setData(dummy, 2, 4, 3);

    EXPECT_EQ(img.width(), 2);
    EXPECT_EQ(img.height(), 4);
    EXPECT_EQ(img.channels(), 3);
    EXPECT_FALSE(img.isHDR());

    auto data_tensor = img.data();

    EXPECT_EQ(data_tensor.device().type(), atcg::CPU);
    EXPECT_EQ(data_tensor.scalar_type(), torch::kUInt8);
    EXPECT_EQ(data_tensor.size(0), 4);
    EXPECT_EQ(data_tensor.size(1), 2);
    EXPECT_EQ(data_tensor.size(2), 3);
}

TEST(ImageTest, rawFloatsetData)
{
    float dummy[2 * 4 * 3];
    atcg::Image img;
    img.setData(dummy, 2, 4, 3);

    EXPECT_EQ(img.width(), 2);
    EXPECT_EQ(img.height(), 4);
    EXPECT_EQ(img.channels(), 3);
    EXPECT_TRUE(img.isHDR());

    auto data_tensor = img.data();

    EXPECT_EQ(data_tensor.device().type(), atcg::CPU);
    EXPECT_EQ(data_tensor.scalar_type(), torch::kFloat32);
    EXPECT_EQ(data_tensor.size(0), 4);
    EXPECT_EQ(data_tensor.size(1), 2);
    EXPECT_EQ(data_tensor.size(2), 3);
}

TEST(ImageTest, tensorUint8setData)
{
    torch::Tensor dummy = torch::zeros({4, 2, 3}, atcg::TensorOptions::uint8HostOptions());
    atcg::Image img;
    img.setData(dummy);

    EXPECT_EQ(img.width(), 2);
    EXPECT_EQ(img.height(), 4);
    EXPECT_EQ(img.channels(), 3);
    EXPECT_FALSE(img.isHDR());

    auto data_tensor = img.data();

    EXPECT_EQ(data_tensor.device().type(), atcg::CPU);
    EXPECT_EQ(data_tensor.scalar_type(), torch::kUInt8);
    EXPECT_EQ(data_tensor.size(0), 4);
    EXPECT_EQ(data_tensor.size(1), 2);
    EXPECT_EQ(data_tensor.size(2), 3);
}

TEST(ImageTest, tensorFloatsetData)
{
    torch::Tensor dummy = torch::zeros({4, 2, 3}, atcg::TensorOptions::floatHostOptions());
    atcg::Image img;
    img.setData(dummy);

    EXPECT_EQ(img.width(), 2);
    EXPECT_EQ(img.height(), 4);
    EXPECT_EQ(img.channels(), 3);
    EXPECT_TRUE(img.isHDR());

    auto data_tensor = img.data();

    EXPECT_EQ(data_tensor.device().type(), atcg::CPU);
    EXPECT_EQ(data_tensor.scalar_type(), torch::kFloat32);
    EXPECT_EQ(data_tensor.size(0), 4);
    EXPECT_EQ(data_tensor.size(1), 2);
    EXPECT_EQ(data_tensor.size(2), 3);
}

TEST(ImageTest, loadLDR)
{
    atcg::Image img;
    img.load((atcg::resource_directory() / "pbr/diffuse.png").string());

    EXPECT_EQ(img.width(), 1024);
    EXPECT_EQ(img.height(), 1024);
    EXPECT_EQ(img.channels(), 3);
    EXPECT_FALSE(img.isHDR());

    auto data_tensor = img.data();

    EXPECT_EQ(data_tensor.device().type(), atcg::CPU);
    EXPECT_EQ(data_tensor.scalar_type(), torch::kUInt8);
    EXPECT_EQ(data_tensor.size(0), 1024);
    EXPECT_EQ(data_tensor.size(1), 1024);
    EXPECT_EQ(data_tensor.size(2), 3);
}

TEST(ImageTest, loadHDR)
{
    atcg::Image img;
    img.load((atcg::resource_directory() / "pbr/skybox.hdr").string());

    EXPECT_EQ(img.width(), 4096);
    EXPECT_EQ(img.height(), 2048);
    EXPECT_EQ(img.channels(), 3);
    EXPECT_TRUE(img.isHDR());

    auto data_tensor = img.data();

    EXPECT_EQ(data_tensor.device().type(), atcg::CPU);
    EXPECT_EQ(data_tensor.scalar_type(), torch::kFloat32);
    EXPECT_EQ(data_tensor.size(0), 2048);
    EXPECT_EQ(data_tensor.size(1), 4096);
    EXPECT_EQ(data_tensor.size(2), 3);
}