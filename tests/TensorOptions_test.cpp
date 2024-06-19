#include <gtest/gtest.h>
#include <Math/Functions.h>
#include <DataStructure/TorchUtils.h>

TEST(TensorOptionsTest, uint8Host)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::uint8HostOptions());

    EXPECT_EQ(t.scalar_type(), torch::kUInt8);
    EXPECT_EQ(t.device().type(), atcg::CPU);
}

TEST(TensorOptionsTest, int8Host)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::int8HostOptions());

    EXPECT_EQ(t.scalar_type(), torch::kInt8);
    EXPECT_EQ(t.device().type(), atcg::CPU);
}

TEST(TensorOptionsTest, int16Host)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::int16HostOptions());

    EXPECT_EQ(t.scalar_type(), torch::kInt16);
    EXPECT_EQ(t.device().type(), atcg::CPU);
}

TEST(TensorOptionsTest, int32Host)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::int32HostOptions());

    EXPECT_EQ(t.scalar_type(), torch::kInt32);
    EXPECT_EQ(t.device().type(), atcg::CPU);
}

TEST(TensorOptionsTest, int64Host)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::int64HostOptions());

    EXPECT_EQ(t.scalar_type(), torch::kInt64);
    EXPECT_EQ(t.device().type(), atcg::CPU);
}

TEST(TensorOptionsTest, floatHost)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::floatHostOptions());

    EXPECT_EQ(t.scalar_type(), torch::kFloat32);
    EXPECT_EQ(t.device().type(), atcg::CPU);
}

TEST(TensorOptionsTest, doubleHost)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::doubleHostOptions());

    EXPECT_EQ(t.scalar_type(), torch::kDouble);
    EXPECT_EQ(t.device().type(), atcg::CPU);
}

TEST(TensorOptionsTest, uint8Device)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::uint8DeviceOptions());

    EXPECT_EQ(t.scalar_type(), torch::kUInt8);
    EXPECT_EQ(t.device().type(), atcg::GPU);
}

TEST(TensorOptionsTest, int8Device)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::int8DeviceOptions());

    EXPECT_EQ(t.scalar_type(), torch::kInt8);
    EXPECT_EQ(t.device().type(), atcg::GPU);
}

TEST(TensorOptionsTest, int16Device)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::int16DeviceOptions());

    EXPECT_EQ(t.scalar_type(), torch::kInt16);
    EXPECT_EQ(t.device().type(), atcg::GPU);
}

TEST(TensorOptionsTest, int32Device)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::int32DeviceOptions());

    EXPECT_EQ(t.scalar_type(), torch::kInt32);
    EXPECT_EQ(t.device().type(), atcg::GPU);
}

TEST(TensorOptionsTest, int64Device)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::int64DeviceOptions());

    EXPECT_EQ(t.scalar_type(), torch::kInt64);
    EXPECT_EQ(t.device().type(), atcg::GPU);
}

TEST(TensorOptionsTest, floatDevice)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::floatDeviceOptions());

    EXPECT_EQ(t.scalar_type(), torch::kFloat32);
    EXPECT_EQ(t.device().type(), atcg::GPU);
}

TEST(TensorOptionsTest, doubleDevice)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::doubleDeviceOptions());

    EXPECT_EQ(t.scalar_type(), torch::kDouble);
    EXPECT_EQ(t.device().type(), atcg::GPU);
}

TEST(TensorOptionsTest, uint8HostOptions)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::HostOptions<uint8_t>());

    EXPECT_EQ(t.scalar_type(), torch::kUInt8);
    EXPECT_EQ(t.device().type(), atcg::CPU);
}

TEST(TensorOptionsTest, int8HostOptions)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::HostOptions<int8_t>());

    EXPECT_EQ(t.scalar_type(), torch::kInt8);
    EXPECT_EQ(t.device().type(), atcg::CPU);
}

TEST(TensorOptionsTest, int16HostOptions)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::HostOptions<int16_t>());

    EXPECT_EQ(t.scalar_type(), torch::kInt16);
    EXPECT_EQ(t.device().type(), atcg::CPU);
}

TEST(TensorOptionsTest, int32HostOptions)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::HostOptions<int32_t>());

    EXPECT_EQ(t.scalar_type(), torch::kInt32);
    EXPECT_EQ(t.device().type(), atcg::CPU);
}

TEST(TensorOptionsTest, int64HostOptions)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::HostOptions<int64_t>());

    EXPECT_EQ(t.scalar_type(), torch::kInt64);
    EXPECT_EQ(t.device().type(), atcg::CPU);
}

TEST(TensorOptionsTest, floatHostOptions)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::HostOptions<float>());

    EXPECT_EQ(t.scalar_type(), torch::kFloat32);
    EXPECT_EQ(t.device().type(), atcg::CPU);
}

TEST(TensorOptionsTest, doubleHostOptions)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::HostOptions<double>());

    EXPECT_EQ(t.scalar_type(), torch::kDouble);
    EXPECT_EQ(t.device().type(), atcg::CPU);
}

TEST(TensorOptionsTest, uint8DeviceOptions)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::DeviceOptions<uint8_t>());

    EXPECT_EQ(t.scalar_type(), torch::kUInt8);
    EXPECT_EQ(t.device().type(), atcg::GPU);
}

TEST(TensorOptionsTest, int8DeviceOptions)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::DeviceOptions<int8_t>());

    EXPECT_EQ(t.scalar_type(), torch::kInt8);
    EXPECT_EQ(t.device().type(), atcg::GPU);
}

TEST(TensorOptionsTest, int16DeviceOptions)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::DeviceOptions<int16_t>());

    EXPECT_EQ(t.scalar_type(), torch::kInt16);
    EXPECT_EQ(t.device().type(), atcg::GPU);
}

TEST(TensorOptionsTest, int32DeviceOptions)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::DeviceOptions<int32_t>());

    EXPECT_EQ(t.scalar_type(), torch::kInt32);
    EXPECT_EQ(t.device().type(), atcg::GPU);
}

TEST(TensorOptionsTest, int64DeviceOptions)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::DeviceOptions<int64_t>());

    EXPECT_EQ(t.scalar_type(), torch::kInt64);
    EXPECT_EQ(t.device().type(), atcg::GPU);
}

TEST(TensorOptionsTest, floatDeviceOptions)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::DeviceOptions<float>());

    EXPECT_EQ(t.scalar_type(), torch::kFloat32);
    EXPECT_EQ(t.device().type(), atcg::GPU);
}

TEST(TensorOptionsTest, doubleDeviceOptions)
{
    torch::Tensor t = torch::zeros({1}, atcg::TensorOptions::DeviceOptions<double>());

    EXPECT_EQ(t.scalar_type(), torch::kDouble);
    EXPECT_EQ(t.device().type(), atcg::GPU);
}