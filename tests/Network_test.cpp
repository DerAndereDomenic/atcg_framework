#include <gtest/gtest.h>
#include <Math/Utils.h>
#include <Core/Platform.h>
#include <Network/NetworkUtils.h>
#include <Network/TCPServer.h>
#include <Network/TCPClient.h>

#include <thread>
#include <chrono>

TEST(NetworkTest, readWriteByte)
{
    uint32_t write_offset = 0;
    uint32_t read_offset  = 0;

    uint8_t* buffer = new uint8_t[1024];

    uint8_t message = 42;

    atcg::NetworkUtils::writeByte(buffer, write_offset, message);
    uint8_t received = atcg::NetworkUtils::readByte(buffer, read_offset);

    EXPECT_EQ(message, received);
    EXPECT_EQ(write_offset, sizeof(uint8_t));
    EXPECT_EQ(read_offset, sizeof(uint8_t));

    delete[] buffer;
}

TEST(NetworkTest, readWriteByteStream)
{
    uint32_t write_offset = 0;
    uint32_t read_offset  = 0;

    uint8_t* buffer = new uint8_t[1024];

    for(int i = 0; i < 1024; ++i)
    {
        uint8_t message = i % 256;
        atcg::NetworkUtils::writeByte(buffer, write_offset, message);
    }

    for(int i = 0; i < 1024; ++i)
    {
        uint8_t message  = i % 256;
        uint8_t received = atcg::NetworkUtils::readByte(buffer, read_offset);
        EXPECT_EQ(message, received);
    }


    delete[] buffer;
}

TEST(NetworkTest, readWriteInt)
{
    uint32_t write_offset = 0;
    uint32_t read_offset  = 0;

    uint8_t* buffer = new uint8_t[1024];

    uint32_t message = 42;

    atcg::NetworkUtils::writeInt(buffer, write_offset, message);
    uint32_t received = atcg::NetworkUtils::readInt<uint32_t>(buffer, read_offset);

    EXPECT_EQ(message, received);
    EXPECT_EQ(write_offset, sizeof(uint32_t));
    EXPECT_EQ(read_offset, sizeof(uint32_t));

    delete[] buffer;
}

TEST(NetworkTest, readWriteIntStream)
{
    uint32_t write_offset = 0;
    uint32_t read_offset  = 0;

    uint32_t* buffer = new uint32_t[1024];

    for(int32_t i = -1024 / 2; i < 1024 / 2; ++i)
    {
        atcg::NetworkUtils::writeInt((uint8_t*)buffer, write_offset, i);
    }

    for(int32_t i = -1024 / 2; i < 1024 / 2; ++i)
    {
        int32_t received = atcg::NetworkUtils::readInt<int32_t>((uint8_t*)buffer, read_offset);
        EXPECT_EQ(i, received);
    }


    delete[] buffer;
}

TEST(NetworkTest, readWriteString)
{
    uint32_t write_offset = 0;
    uint32_t read_offset  = 0;

    uint8_t* buffer = new uint8_t[1024];

    std::string message = "Hello World";

    atcg::NetworkUtils::writeString(buffer, write_offset, message);
    std::string received = atcg::NetworkUtils::readString(buffer, read_offset);

    EXPECT_EQ(message, received);
    EXPECT_EQ(write_offset, sizeof(char) * message.length() + sizeof(uint32_t));
    EXPECT_EQ(read_offset, sizeof(char) * message.length() + sizeof(uint32_t));

    delete[] buffer;
}

TEST(NetworkTest, readWriteStringStream)
{
    uint32_t write_offset = 0;
    uint32_t read_offset  = 0;

    uint8_t* buffer = new uint8_t[1024];

    std::string message1 = "Hallo Welt";
    std::string message2 = "Kannst du mich hören";
    std::string message3 = "Du darfst mich nicht beim chatten stören";

    atcg::NetworkUtils::writeString(buffer, write_offset, message1);
    atcg::NetworkUtils::writeString(buffer, write_offset, message2);
    atcg::NetworkUtils::writeString(buffer, write_offset, message3);
    std::string received1 = atcg::NetworkUtils::readString(buffer, read_offset);
    std::string received2 = atcg::NetworkUtils::readString(buffer, read_offset);
    std::string received3 = atcg::NetworkUtils::readString(buffer, read_offset);

    EXPECT_EQ(message1, received1);
    EXPECT_EQ(message2, received2);
    EXPECT_EQ(message3, received3);
    EXPECT_EQ(write_offset,
              sizeof(char) * (message1.length() + message2.length() + message3.length()) + 3 * sizeof(uint32_t));
    EXPECT_EQ(read_offset,
              sizeof(char) * (message1.length() + message2.length() + message3.length()) + 3 * sizeof(uint32_t));

    delete[] buffer;
}

TEST(NetworkTest, readWriteBuffer)
{
    uint32_t write_offset = 0;
    uint32_t read_offset  = 0;

    uint8_t* buffer  = new uint8_t[1024];
    int num_floats   = 1024 / sizeof(float) - 1;
    float* send_data = new float[num_floats];

    for(int i = 0; i < num_floats; ++i)
    {
        send_data[i] = float(i);
    }

    atcg::NetworkUtils::writeBuffer(buffer, write_offset, (uint8_t*)send_data, 1020);

    uint32_t buffer_size = atcg::NetworkUtils::readInt<uint32_t>(buffer, read_offset);
    EXPECT_EQ(buffer_size, 1020);
    for(int i = 0; i < buffer_size; ++i)
    {
        EXPECT_EQ(((uint8_t*)send_data)[i], buffer[i + read_offset]);
    }

    delete[] buffer;
    delete[] send_data;
}

// 16 bit
TEST(NetworkTest, int16toNetwork)
{
    int16_t local = 0x0102;

    int16_t network = atcg::hton(local);

    int16_t expected = 0x0201;
    if(!atcg::isLittleEndian())
    {
        expected = local;
    }

    EXPECT_EQ(network, expected);
}

TEST(NetworkTest, highint16toNetwork)
{
    int16_t local = 0xff00;

    int16_t network = atcg::hton(local);

    int16_t expected = 0x00ff;
    if(!atcg::isLittleEndian())
    {
        expected = local;
    }

    EXPECT_EQ(network, expected);
}

TEST(NetworkTest, uint16toNetwork)
{
    uint16_t local = 0x0102;

    uint16_t network = atcg::hton(local);

    uint16_t expected = 0x0201;
    if(!atcg::isLittleEndian())
    {
        expected = local;
    }

    EXPECT_EQ(network, expected);
}

TEST(NetworkTest, highuint16toNetwork)
{
    uint16_t local = 0xff00;

    uint16_t network = atcg::hton(local);

    uint16_t expected = 0x00ff;
    if(!atcg::isLittleEndian())
    {
        expected = local;
    }

    EXPECT_EQ(network, expected);
}

TEST(NetworkTest, int16toNetworkRound)
{
    int16_t local = 0x0102;

    int16_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

TEST(NetworkTest, highint16toNetworkRound)
{
    int16_t local = 0x00ff;

    int16_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

TEST(NetworkTest, uint16toNetworkRound)
{
    uint16_t local = 0x0102;

    uint16_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

TEST(NetworkTest, highuint16toNetworkRound)
{
    uint16_t local = 0x00ff;

    uint16_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

// 32 bit
TEST(NetworkTest, int32toNetwork)
{
    int32_t local = 0x01020304;

    int32_t network = atcg::hton(local);

    int32_t expected = 0x04030201;
    if(!atcg::isLittleEndian())
    {
        expected = local;
    }

    EXPECT_EQ(network, expected);
}

TEST(NetworkTest, highint32toNetwork)
{
    int32_t local = 0xff00ff00;

    int32_t network = atcg::hton(local);

    int32_t expected = 0x00ff00ff;
    if(!atcg::isLittleEndian())
    {
        expected = local;
    }

    EXPECT_EQ(network, expected);
}

TEST(NetworkTest, uint32toNetwork)
{
    uint32_t local = 0x01020304;

    uint32_t network = atcg::hton(local);

    uint32_t expected = 0x04030201;
    if(!atcg::isLittleEndian())
    {
        expected = local;
    }

    EXPECT_EQ(network, expected);
}

TEST(NetworkTest, highuint32toNetwork)
{
    uint32_t local = 0xff00ff00;

    uint32_t network = atcg::hton(local);

    uint32_t expected = 0x00ff00ff;
    if(!atcg::isLittleEndian())
    {
        expected = local;
    }

    EXPECT_EQ(network, expected);
}

TEST(NetworkTest, int32toNetworkRound)
{
    int32_t local = 0x01020304;

    int32_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

TEST(NetworkTest, highint32toNetworkRound)
{
    int32_t local = 0x00ff00ff;

    int32_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

TEST(NetworkTest, uint32toNetworkRound)
{
    uint32_t local = 0x01020304;

    uint32_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

TEST(NetworkTest, highuint32toNetworkRound)
{
    uint32_t local = 0x00ff00ff;

    uint32_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

// 64 bit
TEST(NetworkTest, int64toNetwork)
{
    int64_t local = 0x0102030405060708;

    int64_t network = atcg::hton(local);

    int64_t expected = 0x0807060504030201;
    if(!atcg::isLittleEndian())
    {
        expected = local;
    }

    EXPECT_EQ(network, expected);
}

TEST(NetworkTest, highint64toNetwork)
{
    int64_t local = 0xff00ff00ff00ff00;

    int64_t network = atcg::hton(local);

    int64_t expected = 0x00ff00ff00ff00ff;
    if(!atcg::isLittleEndian())
    {
        expected = local;
    }

    EXPECT_EQ(network, expected);
}

TEST(NetworkTest, uint64toNetwork)
{
    uint64_t local = 0x0102030405060708;

    uint64_t network = atcg::hton(local);

    uint64_t expected = 0x0807060504030201;
    if(!atcg::isLittleEndian())
    {
        expected = local;
    }

    EXPECT_EQ(network, expected);
}

TEST(NetworkTest, highuint64toNetwork)
{
    uint64_t local = 0xff00ff00ff00ff00;

    uint64_t network = atcg::hton(local);

    uint64_t expected = 0x00ff00ff00ff00ff;
    if(!atcg::isLittleEndian())
    {
        expected = local;
    }

    EXPECT_EQ(network, expected);
}

TEST(NetworkTest, int64toNetworkRound)
{
    int64_t local = 0x0102030405060708;

    int64_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

TEST(NetworkTest, highint64toNetworkRound)
{
    int64_t local = 0x00ff00ff00ff00ff;

    int64_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

TEST(NetworkTest, uint64toNetworkRound)
{
    uint64_t local = 0x0102030405060708;

    uint64_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

TEST(NetworkTest, highuint64toNetworkRound)
{
    uint64_t local = 0x00ff00ff00ff00ff;

    uint64_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

TEST(NetworkTest, serverStart)
{
    atcg::TCPServer server;
    server.start("127.0.0.1", 25565);
    server.stop();
}

TEST(NetworkTest, connectionTest)
{
    atcg::TCPServer server;
    atcg::TCPClient client;
    server.start("127.0.0.1", 25565);

    client.connect("127.0.0.1", 25565);
    client.disconnect();
    std::this_thread::sleep_for(std::chrono::seconds(1));

    server.stop();
}

TEST(NetworkTest, forceDisconnect)
{
    atcg::TCPServer server;
    atcg::TCPClient client;
    server.start("127.0.0.1", 25565);

    client.connect("127.0.0.1", 25565);

    server.stop();
}

TEST(NetworkTest, simpleEcho)
{
    uint8_t* send_buffer = new uint8_t[1024];

    uint32_t write_offset = 0;
    atcg::NetworkUtils::writeInt<uint32_t>(send_buffer, write_offset, sizeof(uint32_t));
    atcg::NetworkUtils::writeInt<uint32_t>(send_buffer, write_offset, 42);

    atcg::TCPServer server;
    atcg::TCPClient client;
    server.setOnReceiveCallback(
        [&](uint8_t* data, uint32_t data_size, uint64_t client_id)
        {
            uint32_t read_offset = 0;
            uint32_t message     = atcg::NetworkUtils::readInt<uint32_t>(data, read_offset);
            EXPECT_EQ(data_size, sizeof(uint32_t));
            EXPECT_EQ(message, 42);
            server.sendToClient(data, data_size, client_id);
        });
    server.start("127.0.0.1", 25565);

    client.connect("127.0.0.1", 25565);
    auto response = client.sendAndWait(send_buffer, sizeof(uint32_t) * 2);

    uint32_t read_offset = 0;
    EXPECT_EQ(atcg::NetworkUtils::readInt<uint32_t>(response.data_ptr<uint8_t>(), read_offset), 42);

    client.disconnect();
    server.stop();
}

TEST(NetworkTest, streamEcho)
{
    uint8_t* send_buffer = new uint8_t[1024];
    atcg::TCPServer server;
    atcg::TCPClient client;
    server.setOnReceiveCallback([&](uint8_t* data, uint32_t data_size, uint64_t client_id)
                                { server.sendToClient(data, data_size, client_id); });
    server.start("127.0.0.1", 25565);

    client.connect("127.0.0.1", 25565);
    for(uint32_t i = 0; i < 1024; ++i)
    {
        uint32_t write_offset = 0;
        atcg::NetworkUtils::writeInt<uint32_t>(send_buffer, write_offset, sizeof(uint32_t));
        atcg::NetworkUtils::writeInt<uint32_t>(send_buffer, write_offset, i);

        auto response        = client.sendAndWait(send_buffer, sizeof(uint32_t) * 2);
        uint32_t read_offset = 0;
        EXPECT_EQ(atcg::NetworkUtils::readInt<uint32_t>(response.data_ptr<uint8_t>(), read_offset), i);
    }

    client.disconnect();
    server.stop();
}

TEST(NetworkTest, echoLargeData)
{
    int data_size        = 10000;
    auto buffer          = torch::randn({data_size}, atcg::TensorOptions::floatHostOptions());
    uint8_t* send_buffer = new uint8_t[buffer.numel() * buffer.element_size() + sizeof(uint32_t)];

    uint32_t write_offset = 0;
    atcg::NetworkUtils::writeBuffer(send_buffer,
                                    write_offset,
                                    (uint8_t*)buffer.data_ptr(),
                                    buffer.numel() * buffer.element_size());

    atcg::TCPServer server;
    atcg::TCPClient client;
    server.setOnReceiveCallback([&](uint8_t* data, uint32_t data_size, uint64_t client_id)
                                { server.sendToClient(data, data_size, client_id); });
    server.start("127.0.0.1", 25565);

    client.connect("127.0.0.1", 25565);
    auto response = client.sendAndWait(send_buffer, buffer.numel() * buffer.element_size() + sizeof(uint32_t));

    float* float_ptr = reinterpret_cast<float*>(response.data_ptr());

    response = atcg::createHostTensorFromPointer(float_ptr, {data_size}).clone();

    EXPECT_EQ(torch::sum(response - buffer).item<float>(), 0);

    client.disconnect();
    server.stop();
}

TEST(NetworkTest, echoExtraLargeData)
{
    int data_size        = 5000000;
    auto buffer          = torch::randn({data_size}, atcg::TensorOptions::floatHostOptions());
    uint8_t* send_buffer = new uint8_t[buffer.numel() * buffer.element_size() + sizeof(uint32_t)];

    uint32_t write_offset = 0;
    atcg::NetworkUtils::writeBuffer(send_buffer,
                                    write_offset,
                                    (uint8_t*)buffer.data_ptr(),
                                    buffer.numel() * buffer.element_size());

    atcg::TCPServer server;
    atcg::TCPClient client;
    server.setOnReceiveCallback([&](uint8_t* data, uint32_t data_size, uint64_t client_id)
                                { server.sendToClient(data, data_size, client_id); });
    server.start("127.0.0.1", 25565);

    client.connect("127.0.0.1", 25565);
    auto response = client.sendAndWait(send_buffer, buffer.numel() * buffer.element_size() + sizeof(uint32_t));

    float* float_ptr = reinterpret_cast<float*>(response.data_ptr());

    response = atcg::createHostTensorFromPointer(float_ptr, {data_size}).clone();

    EXPECT_EQ(torch::sum(response - buffer).item<float>(), 0);

    client.disconnect();
    server.stop();
}

TEST(NetworkTest, calculateExtraLargeData)
{
    int data_size        = 5000000;
    auto buffer          = torch::randn({data_size}, atcg::TensorOptions::floatHostOptions());
    uint8_t* send_buffer = new uint8_t[buffer.numel() * buffer.element_size() + sizeof(uint32_t)];

    uint32_t write_offset = 0;
    atcg::NetworkUtils::writeBuffer(send_buffer,
                                    write_offset,
                                    (uint8_t*)buffer.data_ptr(),
                                    buffer.numel() * buffer.element_size());

    atcg::TCPServer server;
    atcg::TCPClient client;
    server.setOnReceiveCallback(
        [&](uint8_t* data, uint32_t data_size, uint64_t client_id)
        {
            float* float_ptr    = (float*)data;
            uint32_t num_floats = data_size / sizeof(float);

            for(int i = 0; i < num_floats; ++i)
            {
                float_ptr[i] *= 2.0f;
            }

            server.sendToClient(data, data_size, client_id);
        });
    server.start("127.0.0.1", 25565);

    client.connect("127.0.0.1", 25565);
    auto response = client.sendAndWait(send_buffer, buffer.numel() * buffer.element_size() + sizeof(uint32_t));

    float* float_ptr = reinterpret_cast<float*>(response.data_ptr());

    response = atcg::createHostTensorFromPointer(float_ptr, {data_size}).clone();

    EXPECT_EQ(torch::sum(response - 2.0f * buffer).item<float>(), 0);

    client.disconnect();
    server.stop();
}