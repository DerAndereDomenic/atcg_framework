#include <gtest/gtest.h>
#include <Math/Utils.h>
#include <Core/Platform.h>

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

TEST(EndianTest, highint16toNetwork)
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

TEST(EndianTest, uint16toNetwork)
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

TEST(EndianTest, highuint16toNetwork)
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

TEST(EndianTest, int16toNetworkRound)
{
    int16_t local = 0x0102;

    int16_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

TEST(EndianTest, highint16toNetworkRound)
{
    int16_t local = 0x00ff;

    int16_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

TEST(EndianTest, uint16toNetworkRound)
{
    uint16_t local = 0x0102;

    uint16_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

TEST(EndianTest, highuint16toNetworkRound)
{
    uint16_t local = 0x00ff;

    uint16_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

// 32 bit
TEST(EndianTest, int32toNetwork)
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

TEST(EndianTest, highint32toNetwork)
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

TEST(EndianTest, uint32toNetwork)
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

TEST(EndianTest, highuint32toNetwork)
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

TEST(EndianTest, int32toNetworkRound)
{
    int32_t local = 0x01020304;

    int32_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

TEST(EndianTest, highint32toNetworkRound)
{
    int32_t local = 0x00ff00ff;

    int32_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

TEST(EndianTest, uint32toNetworkRound)
{
    uint32_t local = 0x01020304;

    uint32_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

TEST(EndianTest, highuint32toNetworkRound)
{
    uint32_t local = 0x00ff00ff;

    uint32_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

// 64 bit
TEST(EndianTest, int64toNetwork)
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

TEST(EndianTest, highint64toNetwork)
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

TEST(EndianTest, uint64toNetwork)
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

TEST(EndianTest, highuint64toNetwork)
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

TEST(EndianTest, int64toNetworkRound)
{
    int64_t local = 0x0102030405060708;

    int64_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

TEST(EndianTest, highint64toNetworkRound)
{
    int64_t local = 0x00ff00ff00ff00ff;

    int64_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

TEST(EndianTest, uint64toNetworkRound)
{
    uint64_t local = 0x0102030405060708;

    uint64_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}

TEST(EndianTest, highuint64toNetworkRound)
{
    uint64_t local = 0x00ff00ff00ff00ff;

    uint64_t network = atcg::hton(local);

    EXPECT_EQ(local, atcg::ntoh(network));
}