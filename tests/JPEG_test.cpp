#include <gtest/gtest.h>

#ifdef ATCG_CUDA_BACKEND
    #include <DataStructure/JPEGDecoder.h>
    #include <DataStructure/JPEGENcoder.h>
    #include <DataStructure/Image.h>
    #include <fstream>

TEST(JPEGDecoderTest, decodeFlip)
{
    std::string path = "res/test_data/test.jpeg";
    auto target      = atcg::IO::imread(path);

    auto target_tensor = target->data().flip(0).to(torch::kFloat32) / 255.0f;

    atcg::JPEGDecoder decoder(1, target->width(), target->height(), false);

    std::ifstream input("res/test_data/test.jpeg", std::ios::in | std::ios::binary | std::ios::ate);
    std::streamsize file_size = input.tellg();
    input.seekg(0, std::ios::beg);

    std::vector<std::vector<uint8_t>> file_data(1);

    if(file_data[0].size() < file_size)
    {
        file_data[0].resize(file_size);
    }
    if(!input.read((char*)file_data[0].data(), file_size))
    {
        std::cerr << "JPEGDecoder: Cannot read from file: " << path << "\n";
    }

    auto output = decoder.decompressImages(file_data).to(torch::kFloat32) / 255.0f;

    auto error = torch::mean(torch::square(output.squeeze(0).cpu() - target_tensor));

    EXPECT_NEAR(error.item<float>(), 0.0f, 1e-3f);
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), target->height());
    EXPECT_EQ(output.size(2), target->width());
    EXPECT_EQ(output.size(3), 3);
}

TEST(JPEGDecoderTest, batchDecodeFlip)
{
    int batch_size   = 15;
    std::string path = "res/test_data/test.jpeg";
    auto target      = atcg::IO::imread(path);

    auto target_tensor = target->data().flip(0).to(torch::kFloat32) / 255.0f;

    atcg::JPEGDecoder decoder(batch_size, target->width(), target->height(), false);

    std::vector<std::vector<uint8_t>> file_data(batch_size);
    for(int i = 0; i < batch_size; ++i)
    {
        std::ifstream input("res/test_data/test.jpeg", std::ios::in | std::ios::binary | std::ios::ate);
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);


        if(file_data[i].size() < file_size)
        {
            file_data[i].resize(file_size);
        }
        if(!input.read((char*)file_data[i].data(), file_size))
        {
            std::cerr << "JPEGDecoder: Cannot read from file: " << path << "\n";
        }
    }


    auto output = decoder.decompressImages(file_data).to(torch::kFloat32) / 255.0f;

    auto error = torch::mean(torch::square(output.cpu() - target_tensor.unsqueeze(0)));

    EXPECT_NEAR(error.item<float>(), 0.0f, 1e-3f);
    EXPECT_EQ(output.size(0), batch_size);
    EXPECT_EQ(output.size(1), target->height());
    EXPECT_EQ(output.size(2), target->width());
    EXPECT_EQ(output.size(3), 3);
}

TEST(JPEGDecoderTest, decode)
{
    std::string path = "res/test_data/test.jpeg";
    auto target      = atcg::IO::imread(path);

    auto target_tensor = target->data().to(torch::kFloat32) / 255.0f;

    atcg::JPEGDecoder decoder(1, target->width(), target->height());

    std::ifstream input("res/test_data/test.jpeg", std::ios::in | std::ios::binary | std::ios::ate);
    std::streamsize file_size = input.tellg();
    input.seekg(0, std::ios::beg);

    std::vector<std::vector<uint8_t>> file_data(1);

    if(file_data[0].size() < file_size)
    {
        file_data[0].resize(file_size);
    }
    if(!input.read((char*)file_data[0].data(), file_size))
    {
        std::cerr << "JPEGDecoder: Cannot read from file: " << path << "\n";
    }

    auto output = decoder.decompressImages(file_data).to(torch::kFloat32) / 255.0f;

    auto error = torch::mean(torch::square(output.squeeze(0).cpu() - target_tensor));

    EXPECT_NEAR(error.item<float>(), 0.0f, 1e-3f);
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), target->height());
    EXPECT_EQ(output.size(2), target->width());
    EXPECT_EQ(output.size(3), 3);
}

TEST(JPEGDecoderTest, batchDecode)
{
    int batch_size   = 15;
    std::string path = "res/test_data/test.jpeg";
    auto target      = atcg::IO::imread(path);

    auto target_tensor = target->data().to(torch::kFloat32) / 255.0f;

    atcg::JPEGDecoder decoder(batch_size, target->width(), target->height());

    std::vector<std::vector<uint8_t>> file_data(batch_size);
    for(int i = 0; i < batch_size; ++i)
    {
        std::ifstream input("res/test_data/test.jpeg", std::ios::in | std::ios::binary | std::ios::ate);
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);


        if(file_data[i].size() < file_size)
        {
            file_data[i].resize(file_size);
        }
        if(!input.read((char*)file_data[i].data(), file_size))
        {
            std::cerr << "JPEGDecoder: Cannot read from file: " << path << "\n";
        }
    }


    auto output = decoder.decompressImages(file_data).to(torch::kFloat32) / 255.0f;

    auto error = torch::mean(torch::square(output.cpu() - target_tensor.unsqueeze(0)));

    EXPECT_NEAR(error.item<float>(), 0.0f, 1e-3f);
    EXPECT_EQ(output.size(0), batch_size);
    EXPECT_EQ(output.size(1), target->height());
    EXPECT_EQ(output.size(2), target->width());
    EXPECT_EQ(output.size(3), 3);
}

TEST(JPEGDecoderTest, encode)
{
    auto original        = atcg::IO::imread("res/test_data/test.png");
    auto original_tensor = original->data().cuda();
    auto target          = atcg::IO::imread("res/test_data/test.jpeg");
    auto target_tensor   = target->data().to(torch::kFloat32) / 255.0f;

    atcg::JPEGEncoder encoder;

    torch::Tensor compressed = encoder.compress(original_tensor);

    std::vector<std::vector<uint8_t>> file_data(1);
    file_data[0] =
        std::vector<uint8_t>(compressed.data_ptr<uint8_t>(), compressed.data_ptr<uint8_t>() + compressed.numel());

    atcg::JPEGDecoder decoder(1, original->width(), original->height(), false);
    auto output = decoder.decompressImages(file_data).to(torch::kFloat32) / 255.0f;

    auto error = torch::mean(torch::square(output.cpu() - target_tensor.unsqueeze(0)));

    EXPECT_NEAR(error.item<float>(), 0.0f, 1e-3f);
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), target->height());
    EXPECT_EQ(output.size(2), target->width());
    EXPECT_EQ(output.size(3), 3);
}

#endif