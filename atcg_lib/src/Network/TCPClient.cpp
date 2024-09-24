#include <Network/TCPClient.h>

#include <Core/Assert.h>
#include <DataStructure/TorchUtils.h>
#include <Math/Utils.h>

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

torch::Tensor TCPClient::sendAndWait(uint8_t* data, const uint32_t data_size)
{
    if(impl->socket.send(data, data_size) != sf::Socket::Status::Done)
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

    torch::Tensor rec_data = torch::empty({(int)message_size}, atcg::TensorOptions::uint8HostOptions());

    uint32_t expected_size = atcg::hton<uint32_t>(message_size);

    total_received = 0;
    while(total_received < expected_size)
    {
        impl->socket.receive((char*)((uint8_t*)rec_data.data_ptr() + total_received),
                             rec_data.numel() - total_received,
                             received);
        total_received += received;

        if(total_received >= rec_data.numel())
        {
            // If we overflow our buffer, resize
            rec_data.resize_(2 * rec_data.numel());
        }
    }

    // Zero terminate data
    rec_data.resize_(total_received);
    return rec_data;
}
}    // namespace atcg