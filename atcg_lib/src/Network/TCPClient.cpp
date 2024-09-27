#include <Network/TCPClient.h>

#include <Core/Assert.h>
#include <DataStructure/TorchUtils.h>
#include <Math/Utils.h>
#include <Network/NetworkUtils.h>

namespace atcg
{
class TCPClient::Impl
{
public:
    Impl();
    ~Impl();

    sf::TcpSocket socket;
    bool connected = false;
};

TCPClient::Impl::Impl() {}

TCPClient::Impl::~Impl() {}

TCPClient::TCPClient()
{
    impl = std::make_unique<Impl>();
}

TCPClient::~TCPClient()
{
    if(impl->connected)
    {
        disconnect();
    }
}

void TCPClient::connect(const std::string& ip, const uint32_t port)
{
    ATCG_ASSERT(!impl->connected, "Client already connected");
    if(impl->socket.connect(sf::IpAddress::resolve(ip).value(), port) != sf::Socket::Status::Done)
    {
        ATCG_ERROR("Could not connect to server {0}:{1}. Aborting...", ip, port);
        return;
    }

    impl->connected = true;
}

void TCPClient::disconnect()
{
    ATCG_ASSERT(impl->connected, "Cannot disconnect client without active connection");
    impl->socket.disconnect();
    impl->connected = false;
}

std::vector<uint8_t> TCPClient::sendAndWait(const std::vector<uint8_t>& data)
{
    sf::Packet packet;
    packet.append(data.data(), data.size());
    if(impl->socket.send(packet) != sf::Socket::Status::Done)
    {
        ATCG_ERROR("Could not send data...");
        return {};
    }

    // Wait for response
    std::size_t received;
    std::size_t total_received = 0;

    // First, we need to fetch data until we have received the first 4 bytes -> number of expected bytes
    uint32_t message_size = 0;
    do
    {
        impl->socket.receive((char*)(&message_size), sizeof(uint32_t) - total_received, received);
        total_received += received;
    } while(total_received < sizeof(uint32_t));

    uint32_t expected_size = atcg::hton<uint32_t>(message_size);
    std::vector<uint8_t> rec_data(expected_size);
    total_received = 0;
    while(total_received < expected_size)
    {
        impl->socket.receive((char*)(rec_data.data() + total_received), expected_size - total_received, received);
        total_received += received;
    }

    return rec_data;
}
}    // namespace atcg