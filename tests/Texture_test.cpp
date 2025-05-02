#include <gtest/gtest.h>
#include <Renderer/Texture.h>


TEST(TextureTest, Texture2DcreateEmptyDefault)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    auto texture = atcg::Texture2D::create(spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DcreateEmptyRINT8)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::Texture2D::create(spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DcreateEmptyRG)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::Texture2D::create(spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DcreateEmptyRGB)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::Texture2D::create(spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DcreateEmptyRGBA)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::Texture2D::create(spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DcreateEmptyRFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::Texture2D::create(spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DcreateEmptyRGFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::Texture2D::create(spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DcreateEmptyRGBFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::Texture2D::create(spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DcreateEmptyRGBAFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::Texture2D::create(spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DcreateEmptyRINT)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::Texture2D::create(spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DcreateEmptyDepth)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::DEPTH;
    auto texture = atcg::Texture2D::create(spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DcreateTensorRINT8)
{
    torch::Tensor data = torch::randint(0, 255, {600, 800, 1}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT8);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateTensorRG)
{
    torch::Tensor data = torch::randint(0, 255, {600, 800, 2}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RG);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateTensorRGB)
{
    torch::Tensor data = torch::randint(0, 255, {600, 800, 3}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGB);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateTensorRGBA)
{
    torch::Tensor data = torch::randint(0, 255, {600, 800, 4}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBA);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateTensorRFLOAT)
{
    torch::Tensor data = torch::randn({600, 800, 1}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateTensorRGFLOAT)
{
    torch::Tensor data = torch::randn({600, 800, 2}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateTensorRGBFLOAT)
{
    torch::Tensor data = torch::randn({600, 800, 3}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateTensorRGBAFLOAT)
{
    torch::Tensor data = torch::randn({600, 800, 4}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBAFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateTensorRINT)
{
    torch::Tensor data = torch::randint(0, 1e6, {600, 800, 1}, atcg::TensorOptions::int32HostOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kInt32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateTensorRINT8GPU)
{
    torch::Tensor data = torch::randint(0, 255, {600, 800, 1}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT8);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateTensorRGGPU)
{
    torch::Tensor data = torch::randint(0, 255, {600, 800, 2}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RG);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateTensorRGBGPU)
{
    torch::Tensor data = torch::randint(0, 255, {600, 800, 3}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGB);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateTensorRGBAGPU)
{
    torch::Tensor data = torch::randint(0, 255, {600, 800, 4}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBA);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateTensorRFLOATGPU)
{
    torch::Tensor data = torch::randn({600, 800, 1}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateTensorRGFLOATGPU)
{
    torch::Tensor data = torch::randn({600, 800, 2}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateTensorRGBFLOATGPU)
{
    torch::Tensor data = torch::randn({600, 800, 3}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateTensorRGBAFLOATGPU)
{
    torch::Tensor data = torch::randn({600, 800, 4}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBAFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateTensorRINTGPU)
{
    torch::Tensor data = torch::randint(0, 1e6, {600, 800, 1}, atcg::TensorOptions::int32DeviceOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kInt32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateTensorRINT8getGPU)
{
    torch::Tensor data = torch::randint(0, 255, {600, 800, 1}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT8);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateTensorRGgetGPU)
{
    torch::Tensor data = torch::randint(0, 255, {600, 800, 2}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RG);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateTensorRGBgetGPU)
{
    torch::Tensor data = torch::randint(0, 255, {600, 800, 3}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGB);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateTensorRGBAgetGPU)
{
    torch::Tensor data = torch::randint(0, 255, {600, 800, 4}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBA);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateTensorRFLOATgetGPU)
{
    torch::Tensor data = torch::randn({600, 800, 1}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateTensorRGFLOATgetGPU)
{
    torch::Tensor data = torch::randn({600, 800, 2}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateTensorRGBFLOATgetGPU)
{
    torch::Tensor data = torch::randn({600, 800, 3}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateTensorRGBAFLOATgetGPU)
{
    torch::Tensor data = torch::randn({600, 800, 4}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBAFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateTensorRINTgetGPU)
{
    torch::Tensor data = torch::randint(0, 1e6, {600, 800, 1}, atcg::TensorOptions::int32HostOptions());
    auto texture       = atcg::Texture2D::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kInt32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRINT8setDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 1}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRGsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 2}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRGBsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 3}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRGBAsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 4}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 1}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRGFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 2}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRGBFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 3}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRGBAFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 4}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRINTsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 1e6, {600, 800, 1}, atcg::TensorOptions::int32HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRINT8setDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 1}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRGsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 2}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRGBsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 3}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRGBAsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 4}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 1}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRGFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 2}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRGBFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 3}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRGBAFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 4}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRINTsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 1e6, {600, 800, 1}, atcg::TensorOptions::int32DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRINT8setDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 1}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRGsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 2}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRGBsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 3}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRGBAsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 4}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 1}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRGFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 2}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRGBFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 3}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRGBAFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {600, 800, 4}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture2DcreateEmptyRINTsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::Texture2D::create(spec);

    torch::Tensor data = torch::randint(0, 1e6, {600, 800, 1}, atcg::TensorOptions::int32HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

TEST(TextureTest, Texture3DcreateEmptyDefault)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    auto texture = atcg::Texture3D::create(spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture3DcreateEmptyRINT8)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::Texture3D::create(spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture3DcreateEmptyRG)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::Texture3D::create(spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture3DcreateEmptyRGB)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::Texture3D::create(spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture3DcreateEmptyRGBA)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::Texture3D::create(spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture3DcreateEmptyRFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::Texture3D::create(spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture3DcreateEmptyRGFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::Texture3D::create(spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture3DcreateEmptyRGBFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::Texture3D::create(spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture3DcreateEmptyRGBAFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::Texture3D::create(spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture3DcreateEmptyRINT)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::Texture3D::create(spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture3DcreateTensorRINT8)
{
    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 1}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT8);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateTensorRG)
{
    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 2}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RG);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateTensorRGB)
{
    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 3}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGB);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateTensorRGBA)
{
    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 4}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBA);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateTensorRFLOAT)
{
    torch::Tensor data = torch::randn({5, 600, 800, 1}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateTensorRGFLOAT)
{
    torch::Tensor data = torch::randn({5, 600, 800, 2}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateTensorRGBFLOAT)
{
    torch::Tensor data = torch::randn({5, 600, 800, 3}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateTensorRGBAFLOAT)
{
    torch::Tensor data = torch::randn({5, 600, 800, 4}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBAFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateTensorRINT)
{
    torch::Tensor data = torch::randint(0, 1e6, {5, 600, 800, 1}, atcg::TensorOptions::int32HostOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kInt32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateTensorRINT8GPU)
{
    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 1}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT8);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateTensorRGGPU)
{
    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 2}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RG);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateTensorRGBGPU)
{
    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 3}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGB);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateTensorRGBAGPU)
{
    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 4}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBA);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateTensorRFLOATGPU)
{
    torch::Tensor data = torch::randn({5, 600, 800, 1}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateTensorRGFLOATGPU)
{
    torch::Tensor data = torch::randn({5, 600, 800, 2}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateTensorRGBFLOATGPU)
{
    torch::Tensor data = torch::randn({5, 600, 800, 3}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateTensorRGBAFLOATGPU)
{
    torch::Tensor data = torch::randn({5, 600, 800, 4}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBAFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateTensorRINTGPU)
{
    torch::Tensor data = torch::randint(0, 1e6, {5, 600, 800, 1}, atcg::TensorOptions::int32DeviceOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kInt32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateTensorRINT8getGPU)
{
    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 1}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT8);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateTensorRGgetGPU)
{
    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 2}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RG);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateTensorRGBgetGPU)
{
    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 3}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGB);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateTensorRGBAgetGPU)
{
    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 4}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBA);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateTensorRFLOATgetGPU)
{
    torch::Tensor data = torch::randn({5, 600, 800, 1}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateTensorRGFLOATgetGPU)
{
    torch::Tensor data = torch::randn({5, 600, 800, 2}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateTensorRGBFLOATgetGPU)
{
    torch::Tensor data = torch::randn({5, 600, 800, 3}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateTensorRGBAFLOATgetGPU)
{
    torch::Tensor data = torch::randn({5, 600, 800, 4}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBAFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateTensorRINTgetGPU)
{
    torch::Tensor data = torch::randint(0, 1e6, {5, 600, 800, 1}, atcg::TensorOptions::int32HostOptions());
    auto texture       = atcg::Texture3D::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kInt32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRINT8setDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 1}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRGsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 2}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRGBsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 3}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRGBAsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 4}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 1}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRGFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 2}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRGBFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 3}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRGBAFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 4}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRINTsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 1e6, {5, 600, 800, 1}, atcg::TensorOptions::int32HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRINT8setDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 1}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRGsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 2}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRGBsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 3}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRGBAsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 4}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 1}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRGFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 2}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRGBFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 3}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRGBAFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 4}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRINTsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 1e6, {5, 600, 800, 1}, atcg::TensorOptions::int32DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRINT8setDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 1}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRGsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 2}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRGBsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 3}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRGBAsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 4}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 1}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRGFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 2}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRGBFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 3}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRGBAFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 255, {5, 600, 800, 4}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, Texture3DcreateEmptyRINTsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.depth   = 5;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::Texture3D::create(spec);

    torch::Tensor data = torch::randint(0, 1e6, {5, 600, 800, 1}, atcg::TensorOptions::int32HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 5);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------

TEST(TextureTest, TextureCubecreateEmptyDefault)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    auto texture = atcg::TextureCube::create(spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, TextureCubecreateEmptyRINT8)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::TextureCube::create(spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, TextureCubecreateEmptyRG)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::TextureCube::create(spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, TextureCubecreateEmptyRGB)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::TextureCube::create(spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, TextureCubecreateEmptyRGBA)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::TextureCube::create(spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, TextureCubecreateEmptyRFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::TextureCube::create(spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, TextureCubecreateEmptyRGFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::TextureCube::create(spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, TextureCubecreateEmptyRGBFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::TextureCube::create(spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, TextureCubecreateEmptyRGBAFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::TextureCube::create(spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, TextureCubecreateEmptyRINT)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::TextureCube::create(spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, TextureCubecreateTensorRINT8)
{
    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 1}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT8);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateTensorRG)
{
    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 2}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RG);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateTensorRGB)
{
    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 3}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGB);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateTensorRGBA)
{
    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 4}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBA);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateTensorRFLOAT)
{
    torch::Tensor data = torch::randn({6, 600, 600, 1}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateTensorRGFLOAT)
{
    torch::Tensor data = torch::randn({6, 600, 600, 2}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateTensorRGBFLOAT)
{
    torch::Tensor data = torch::randn({6, 600, 600, 3}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateTensorRGBAFLOAT)
{
    torch::Tensor data = torch::randn({6, 600, 600, 4}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBAFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateTensorRINT)
{
    torch::Tensor data = torch::randint(0, 1e6, {6, 600, 600, 1}, atcg::TensorOptions::int32HostOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kInt32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateTensorRINT8GPU)
{
    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 1}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT8);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateTensorRGGPU)
{
    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 2}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RG);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateTensorRGBGPU)
{
    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 3}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGB);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateTensorRGBAGPU)
{
    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 4}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBA);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateTensorRFLOATGPU)
{
    torch::Tensor data = torch::randn({6, 600, 600, 1}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateTensorRGFLOATGPU)
{
    torch::Tensor data = torch::randn({6, 600, 600, 2}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateTensorRGBFLOATGPU)
{
    torch::Tensor data = torch::randn({6, 600, 600, 3}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateTensorRGBAFLOATGPU)
{
    torch::Tensor data = torch::randn({6, 600, 600, 4}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBAFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateTensorRINTGPU)
{
    torch::Tensor data = torch::randint(0, 1e6, {6, 600, 600, 1}, atcg::TensorOptions::int32DeviceOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kInt32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateTensorRINT8getGPU)
{
    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 1}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT8);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateTensorRGgetGPU)
{
    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 2}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RG);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateTensorRGBgetGPU)
{
    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 3}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGB);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateTensorRGBAgetGPU)
{
    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 4}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBA);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateTensorRFLOATgetGPU)
{
    torch::Tensor data = torch::randn({6, 600, 600, 1}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateTensorRGFLOATgetGPU)
{
    torch::Tensor data = torch::randn({6, 600, 600, 2}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateTensorRGBFLOATgetGPU)
{
    torch::Tensor data = torch::randn({6, 600, 600, 3}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateTensorRGBAFLOATgetGPU)
{
    torch::Tensor data = torch::randn({6, 600, 600, 4}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBAFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateTensorRINTgetGPU)
{
    torch::Tensor data = torch::randint(0, 1e6, {6, 600, 600, 1}, atcg::TensorOptions::int32HostOptions());
    auto texture       = atcg::TextureCube::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kInt32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRINT8setDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 1}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRGsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 2}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRGBsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 3}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRGBAsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 4}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 1}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRGFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 2}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRGBFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 3}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRGBAFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 4}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRINTsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 1e6, {6, 600, 600, 1}, atcg::TensorOptions::int32HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRINT8setDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 1}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRGsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 2}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRGBsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 3}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRGBAsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 4}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 1}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRGFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 2}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRGBFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 3}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRGBAFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 4}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRINTsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 1e6, {6, 600, 600, 1}, atcg::TensorOptions::int32DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRINT8setDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 1}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRGsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 2}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRGBsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 3}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRGBAsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 4}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 1}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRGFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 2}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRGBFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 3}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRGBAFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 255, {6, 600, 600, 4}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubecreateEmptyRINTsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::TextureCube::create(spec);

    torch::Tensor data = torch::randint(0, 1e6, {6, 600, 600, 1}, atcg::TensorOptions::int32HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------

TEST(TextureTest, TextureArraycreateEmptyDefault)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    auto texture = atcg::TextureArray::create(spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, TextureArraycreateEmptyRINT8)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::TextureArray::create(spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, TextureArraycreateEmptyRG)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::TextureArray::create(spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, TextureArraycreateEmptyRGB)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::TextureArray::create(spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, TextureArraycreateEmptyRGBA)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::TextureArray::create(spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, TextureArraycreateEmptyRFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::TextureArray::create(spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, TextureArraycreateEmptyRGFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::TextureArray::create(spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, TextureArraycreateEmptyRGBFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::TextureArray::create(spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, TextureArraycreateEmptyRGBAFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::TextureArray::create(spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, TextureArraycreateEmptyRINT)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::TextureArray::create(spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, TextureArraycreateTensorRINT8)
{
    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 1}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT8);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateTensorRG)
{
    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 2}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RG);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateTensorRGB)
{
    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 3}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGB);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateTensorRGBA)
{
    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 4}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBA);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateTensorRFLOAT)
{
    torch::Tensor data = torch::randn({10, 600, 600, 1}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateTensorRGFLOAT)
{
    torch::Tensor data = torch::randn({10, 600, 600, 2}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateTensorRGBFLOAT)
{
    torch::Tensor data = torch::randn({10, 600, 600, 3}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateTensorRGBAFLOAT)
{
    torch::Tensor data = torch::randn({10, 600, 600, 4}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBAFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateTensorRINT)
{
    torch::Tensor data = torch::randint(0, 1e6, {10, 600, 600, 1}, atcg::TensorOptions::int32HostOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kInt32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateTensorRINT8GPU)
{
    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 1}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT8);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateTensorRGGPU)
{
    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 2}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RG);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateTensorRGBGPU)
{
    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 3}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGB);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateTensorRGBAGPU)
{
    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 4}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBA);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateTensorRFLOATGPU)
{
    torch::Tensor data = torch::randn({10, 600, 600, 1}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateTensorRGFLOATGPU)
{
    torch::Tensor data = torch::randn({10, 600, 600, 2}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateTensorRGBFLOATGPU)
{
    torch::Tensor data = torch::randn({10, 600, 600, 3}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateTensorRGBAFLOATGPU)
{
    torch::Tensor data = torch::randn({10, 600, 600, 4}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBAFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateTensorRINTGPU)
{
    torch::Tensor data = torch::randint(0, 1e6, {10, 600, 600, 1}, atcg::TensorOptions::int32DeviceOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kInt32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateTensorRINT8getGPU)
{
    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 1}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT8);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateTensorRGgetGPU)
{
    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 2}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RG);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateTensorRGBgetGPU)
{
    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 3}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGB);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateTensorRGBAgetGPU)
{
    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 4}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBA);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateTensorRFLOATgetGPU)
{
    torch::Tensor data = torch::randn({10, 600, 600, 1}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateTensorRGFLOATgetGPU)
{
    torch::Tensor data = torch::randn({10, 600, 600, 2}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateTensorRGBFLOATgetGPU)
{
    torch::Tensor data = torch::randn({10, 600, 600, 3}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateTensorRGBAFLOATgetGPU)
{
    torch::Tensor data = torch::randn({10, 600, 600, 4}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBAFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateTensorRINTgetGPU)
{
    torch::Tensor data = torch::randint(0, 1e6, {10, 600, 600, 1}, atcg::TensorOptions::int32HostOptions());
    auto texture       = atcg::TextureArray::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kInt32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRINT8setDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 1}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRGsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 2}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRGBsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 3}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRGBAsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 4}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 1}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRGFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 2}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRGBFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 3}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRGBAFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 4}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRINTsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 1e6, {10, 600, 600, 1}, atcg::TensorOptions::int32HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRINT8setDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 1}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRGsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 2}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRGBsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 3}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRGBAsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 4}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 1}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRGFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 2}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRGBFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 3}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRGBAFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 4}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRINTsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 1e6, {10, 600, 600, 1}, atcg::TensorOptions::int32DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRINT8setDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 1}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRGsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 2}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRGBsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 3}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRGBAsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 4}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 1}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRGFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 2}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRGBFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 3}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRGBAFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 600, 600, 4}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureArraycreateEmptyRINTsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::TextureArray::create(spec);

    torch::Tensor data = torch::randint(0, 1e6, {10, 600, 600, 1}, atcg::TensorOptions::int32HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------

TEST(TextureTest, TextureCubeArraycreateEmptyDefault)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    auto texture = atcg::TextureCubeArray::create(spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, TextureCubeArraycreateEmptyRINT8)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::TextureCubeArray::create(spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, TextureCubeArraycreateEmptyRG)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::TextureCubeArray::create(spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGB)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::TextureCubeArray::create(spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGBA)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::TextureCubeArray::create(spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, TextureCubeArraycreateEmptyRFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::TextureCubeArray::create(spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::TextureCubeArray::create(spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGBFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::TextureCubeArray::create(spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGBAFLOAT)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::TextureCubeArray::create(spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, TextureCubeArraycreateEmptyRINT)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::TextureCubeArray::create(spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, TextureCubeArraycreateTensorRINT8)
{
    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 1}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT8);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRG)
{
    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 2}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RG);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRGB)
{
    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 3}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGB);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRGBA)
{
    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 4}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBA);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRFLOAT)
{
    torch::Tensor data = torch::randn({10, 6, 600, 600, 1}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRGFLOAT)
{
    torch::Tensor data = torch::randn({10, 6, 600, 600, 2}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRGBFLOAT)
{
    torch::Tensor data = torch::randn({10, 6, 600, 600, 3}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRGBAFLOAT)
{
    torch::Tensor data = torch::randn({10, 6, 600, 600, 4}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBAFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRINT)
{
    torch::Tensor data = torch::randint(0, 1e6, {10, 6, 600, 600, 1}, atcg::TensorOptions::int32HostOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kInt32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRINT8GPU)
{
    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 1}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT8);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRGGPU)
{
    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 2}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RG);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRGBGPU)
{
    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 3}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGB);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRGBAGPU)
{
    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 4}, atcg::TensorOptions::uint8DeviceOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBA);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRFLOATGPU)
{
    torch::Tensor data = torch::randn({10, 6, 600, 600, 1}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRGFLOATGPU)
{
    torch::Tensor data = torch::randn({10, 6, 600, 600, 2}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRGBFLOATGPU)
{
    torch::Tensor data = torch::randn({10, 6, 600, 600, 3}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRGBAFLOATGPU)
{
    torch::Tensor data = torch::randn({10, 6, 600, 600, 4}, atcg::TensorOptions::floatDeviceOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBAFLOAT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRINTGPU)
{
    torch::Tensor data = torch::randint(0, 1e6, {10, 6, 600, 600, 1}, atcg::TensorOptions::int32DeviceOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT);
    EXPECT_EQ(texture->getData(atcg::GPU).scalar_type(), torch::kInt32);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRINT8getGPU)
{
    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 1}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT8);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRGgetGPU)
{
    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 2}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RG);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRGBgetGPU)
{
    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 3}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGB);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRGBAgetGPU)
{
    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 4}, atcg::TensorOptions::uint8HostOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBA);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kUInt8);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRFLOATgetGPU)
{
    torch::Tensor data = torch::randn({10, 6, 600, 600, 1}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRGFLOATgetGPU)
{
    torch::Tensor data = torch::randn({10, 6, 600, 600, 2}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRGBFLOATgetGPU)
{
    torch::Tensor data = torch::randn({10, 6, 600, 600, 3}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRGBAFLOATgetGPU)
{
    torch::Tensor data = torch::randn({10, 6, 600, 600, 4}, atcg::TensorOptions::floatHostOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RGBAFLOAT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateTensorRINTgetGPU)
{
    torch::Tensor data = torch::randint(0, 1e6, {10, 6, 600, 600, 1}, atcg::TensorOptions::int32HostOptions());
    auto texture       = atcg::TextureCubeArray::create(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_EQ(texture->getSpecification().format, atcg::TextureFormat::RINT);
    EXPECT_EQ(texture->getData(atcg::CPU).scalar_type(), torch::kInt32);
    EXPECT_TRUE(torch::allclose(data.to(atcg::GPU), texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRINT8setDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 1}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 2}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGBsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 3}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGBAsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 4}, atcg::TensorOptions::uint8HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 1}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 2}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGBFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 3}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGBAFLOATsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 4}, atcg::TensorOptions::floatHostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRINTsetDataHost)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 1e6, {10, 6, 600, 600, 1}, atcg::TensorOptions::int32HostOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRINT8setDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 1}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 2}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGBsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 3}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGBAsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 4}, atcg::TensorOptions::uint8DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 1}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 2}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGBFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 3}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGBAFLOATsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 4}, atcg::TensorOptions::floatDeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRINTsetDataDevice)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 1e6, {10, 6, 600, 600, 1}, atcg::TensorOptions::int32DeviceOptions());
    texture->setData(data);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::GPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRINT8setDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 1}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 2}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGBsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 3}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGBAsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 4}, atcg::TensorOptions::uint8HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 1}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 2}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGBFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 3}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRGBAFLOATsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 255, {10, 6, 600, 600, 4}, atcg::TensorOptions::floatHostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), true);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

TEST(TextureTest, TextureCubeArraycreateEmptyRINTsetDataPBO)
{
    atcg::TextureSpecification spec;
    spec.width   = 600;
    spec.height  = 600;
    spec.depth   = 10;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::TextureCubeArray::create(spec);

    torch::Tensor data = torch::randint(0, 1e6, {10, 6, 600, 600, 1}, atcg::TensorOptions::int32HostOptions());
    auto pbo           = atcg::make_ref<atcg::PixelUnpackBuffer>(data.data_ptr(), data.numel() * data.element_size());

    texture->setData(pbo);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 600);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 10);
    EXPECT_EQ(texture->isHDR(), false);
    EXPECT_TRUE(torch::allclose(data, texture->getData(atcg::CPU)));
}

//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------

TEST(TextureTest, Texture2DMultiSamplecreateEmptyDefault1Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    auto texture = atcg::Texture2DMultiSample::create(1, spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRINT81Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::Texture2DMultiSample::create(1, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRG1Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::Texture2DMultiSample::create(1, spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGB1Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::Texture2DMultiSample::create(1, spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGBA1Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::Texture2DMultiSample::create(1, spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRFLOAT1Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(1, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGFLOAT1Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(1, spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGBFLOAT1Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(1, spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGBAFLOAT1Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(1, spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRINT1Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::Texture2DMultiSample::create(1, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyDepth1Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::DEPTH;
    auto texture = atcg::Texture2DMultiSample::create(1, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyDefault2Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    auto texture = atcg::Texture2DMultiSample::create(2, spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRINT82Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::Texture2DMultiSample::create(2, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRG2Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::Texture2DMultiSample::create(2, spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGB2Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::Texture2DMultiSample::create(2, spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGBA2Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::Texture2DMultiSample::create(2, spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRFLOAT2Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(2, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGFLOAT2Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(2, spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGBFLOAT2Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(2, spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGBAFLOAT2Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(2, spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRINT2Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::Texture2DMultiSample::create(2, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyDepth2Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::DEPTH;
    auto texture = atcg::Texture2DMultiSample::create(2, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyDefault4Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    auto texture = atcg::Texture2DMultiSample::create(4, spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRINT84Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::Texture2DMultiSample::create(4, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRG4Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::Texture2DMultiSample::create(4, spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGB4Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::Texture2DMultiSample::create(4, spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGBA4Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::Texture2DMultiSample::create(4, spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRFLOAT4Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(4, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGFLOAT4Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(4, spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGBFLOAT4Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(4, spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGBAFLOAT4Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(4, spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRINT4Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::Texture2DMultiSample::create(4, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyDepth4Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::DEPTH;
    auto texture = atcg::Texture2DMultiSample::create(4, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyDefault8Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    auto texture = atcg::Texture2DMultiSample::create(8, spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRINT88Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::Texture2DMultiSample::create(8, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRG8Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::Texture2DMultiSample::create(8, spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGB8Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::Texture2DMultiSample::create(8, spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGBA8Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::Texture2DMultiSample::create(8, spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRFLOAT8Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(8, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGFLOAT8Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(8, spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGBFLOAT8Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(8, spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGBAFLOAT8Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(8, spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRINT8Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::Texture2DMultiSample::create(8, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyDepth8Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::DEPTH;
    auto texture = atcg::Texture2DMultiSample::create(8, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyDefault16Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    auto texture = atcg::Texture2DMultiSample::create(16, spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRINT816Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::Texture2DMultiSample::create(16, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRG16Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::Texture2DMultiSample::create(16, spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGB16Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::Texture2DMultiSample::create(16, spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGBA16Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::Texture2DMultiSample::create(16, spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRFLOAT16Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(16, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGFLOAT16Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(16, spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGBFLOAT16Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(16, spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGBAFLOAT16Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(16, spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRINT16Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::Texture2DMultiSample::create(1, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyDepth16Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::DEPTH;
    auto texture = atcg::Texture2DMultiSample::create(16, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyDefault32Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    auto texture = atcg::Texture2DMultiSample::create(32, spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRINT832Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT8;
    auto texture = atcg::Texture2DMultiSample::create(32, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRG32Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RG;
    auto texture = atcg::Texture2DMultiSample::create(32, spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGB32Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGB;
    auto texture = atcg::Texture2DMultiSample::create(32, spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGBA32Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBA;
    auto texture = atcg::Texture2DMultiSample::create(32, spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRFLOAT32Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(32, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGFLOAT32Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(32, spec);

    EXPECT_EQ(texture->channels(), 2);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGBFLOAT32Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(32, spec);

    EXPECT_EQ(texture->channels(), 3);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRGBAFLOAT32Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RGBAFLOAT;
    auto texture = atcg::Texture2DMultiSample::create(32, spec);

    EXPECT_EQ(texture->channels(), 4);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), true);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyRINT32Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::RINT;
    auto texture = atcg::Texture2DMultiSample::create(32, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}

TEST(TextureTest, Texture2DMultiSamplecreateEmptyDepth32Sample)
{
    atcg::TextureSpecification spec;
    spec.width   = 800;
    spec.height  = 600;
    spec.format  = atcg::TextureFormat::DEPTH;
    auto texture = atcg::Texture2DMultiSample::create(32, spec);

    EXPECT_EQ(texture->channels(), 1);
    EXPECT_EQ(texture->width(), 800);
    EXPECT_EQ(texture->height(), 600);
    EXPECT_EQ(texture->depth(), 0);
    EXPECT_EQ(texture->isHDR(), false);
}