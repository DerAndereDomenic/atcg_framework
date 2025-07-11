#include <Network/TCPServer.h>

#include <unordered_map>
#include <thread>
#include <mutex>

#include <Core/Memory.h>
#include <Core/UUID.h>
#include <DataStructure/TorchUtils.h>
#include <Network/NetworkUtils.h>

#include <DataStructure/Timer.h>


namespace atcg
{
class TCPServer::Impl
{
public:
    Impl();

    ~Impl();

    std::string ip;
    uint32_t port;
    sf::TcpListener listener;
    sf::SocketSelector selector;

    std::mutex queue_mutex;
    std::unordered_map<UUID, atcg::ref_ptr<sf::TcpSocket>> sockets;

    std::atomic_bool running = false;

    std::thread network_thread;
    void networkLoop();

    void pushToQueue(const std::vector<uint8_t>& message, const uint64_t client_id);

    // Only log per default
    NetworkConnectionCallbackFn connect_callback = [&](const uint64_t client_id)
    {
        auto& client = sockets[client_id];
        auto ip      = client->getRemoteAddress().value().toString();
        ATCG_TRACE("Client {0}:{1} connected", ip, client->getRemotePort());
    };
    NetworkConnectionCallbackFn disconnect_callback = [&](const uint64_t client_id)
    {
        auto& client = sockets[client_id];
        auto ip      = client->getRemoteAddress().value().toString();
        ATCG_TRACE("Client {0}:{1} disconnected", ip, client->getRemotePort());
    };
    NetworkCallbackFn receive_callback = [&](std::vector<uint8_t>& message, const uint64_t client_id)
    {
        auto& client = sockets[client_id];
        auto ip      = client->getRemoteAddress().value().toString();
        ATCG_TRACE("Client {0}:{1} received {2} bytes of data", ip, client->getRemotePort(), message.size());

        // Echo message
        pushToQueue(message, client_id);
    };

    struct DataPacket
    {
        DataPacket() = default;

        DataPacket(const std::vector<uint8_t>& message, const uint64_t client_id) : data(message), client_id(client_id)
        {
        }

        std::vector<uint8_t> data;
        uint64_t client_id;
    };

    std::queue<std::unique_ptr<DataPacket>> data_to_process;

    std::vector<uint8_t> rec_data;
};

TCPServer::Impl::Impl() {}

TCPServer::Impl::~Impl() {}

void TCPServer::Impl::pushToQueue(const std::vector<uint8_t>& message, const uint64_t client_id)
{
    std::lock_guard lock(queue_mutex);

    data_to_process.push(std::make_unique<DataPacket>(message, client_id));
}

void TCPServer::Impl::networkLoop()
{
    listener.listen(port);
    selector.add(listener);
    ATCG_TRACE("Server listening on {0}", port);

    // TODO: Make this loop better maintainable...
    while(running)
    {
        if(selector.wait(sf::milliseconds(1.0f)))
        {
            // Handle new incoming connections
            if(selector.isReady(listener))
            {
                atcg::ref_ptr<sf::TcpSocket> client = atcg::make_ref<sf::TcpSocket>();
                if(listener.accept(*client) == sf::Socket::Status::Done)
                {
                    UUID uuid;
                    sockets.insert(std::make_pair(uuid, client));
                    selector.add(*client);

                    connect_callback(uuid);
                }
            }
            else
            {
                // Handle clients
                for(auto it = sockets.begin(); it != sockets.end();)
                {
                    auto& client = it->second;
                    if(selector.isReady(*client))
                    {
                        std::size_t received;
                        std::size_t total_received = 0;

                        uint32_t message_size = 0;
                        bool disconnected     = false;
                        // First, we need to fetch data until we have received the first 4 bytes -> number of expected
                        // bytes
                        do
                        {
                            auto state = client->receive((char*)(&message_size) + total_received,
                                                         sizeof(uint32_t) - total_received,
                                                         received);
                            total_received += received;

                            if(state == sf::Socket::Status::Disconnected)
                            {
                                disconnect_callback(it->first);
                                selector.remove(*client);
                                client->disconnect();

                                it           = sockets.erase(it);
                                disconnected = true;
                                break;
                            }
                        } while(total_received < sizeof(uint32_t));

                        if(disconnected) continue;

                        uint32_t read_offset   = 0;
                        uint32_t expected_size = atcg::ntoh(message_size);
                        rec_data.resize(expected_size);

                        // Do not count header as part of the message
                        total_received = 0;
                        while(total_received < expected_size)
                        {
                            auto status = client->receive((char*)(rec_data.data() + total_received),
                                                          expected_size - total_received,
                                                          received);

                            total_received += received;
                        }

                        receive_callback(rec_data, it->first);
                    }
                    ++it;
                }
            }
        }

        // Check for data to send
        std::unique_ptr<DataPacket> data = nullptr;
        {
            std::lock_guard guard(queue_mutex);

            if(!data_to_process.empty())
            {
                data = std::move(data_to_process.front());
                data_to_process.pop();
            }
        }

        if(data)
        {
            auto& client = sockets[data->client_id];
            sf::Packet packet;
            packet.append(data->data.data(), data->data.size());

            if(client->send(packet) != sf::Socket::Status::Done)
            {
                ATCG_ERROR("Request failed!");
            }
        }
    }

    ATCG_TRACE("Server closing...");
    selector.clear();

    for(auto it = sockets.begin(); it != sockets.end(); ++it)
    {
        it->second->disconnect();
    }
    sockets.clear();
}

TCPServer::TCPServer()
{
    impl = std::make_unique<Impl>();
}

TCPServer::~TCPServer() {}

void TCPServer::start(const std::string& ip, const uint32_t port)
{
    impl->ip   = ip;
    impl->port = port;
    if(!impl->running)
    {
        impl->running        = true;
        impl->network_thread = std::thread(&TCPServer::Impl::networkLoop, impl.get());
    }
}

void TCPServer::stop()
{
    impl->running = false;
    if(impl->network_thread.joinable()) impl->network_thread.join();
}

void TCPServer::setOnConnectCallback(const NetworkConnectionCallbackFn& callback)
{
    impl->connect_callback = callback;
}

void TCPServer::setOnReceiveCallback(const NetworkCallbackFn& callback)
{
    impl->receive_callback = callback;
}

void TCPServer::setOnDisconnectCallback(const NetworkConnectionCallbackFn& callback)
{
    impl->disconnect_callback = callback;
}

void TCPServer::sendToClient(const std::vector<uint8_t>& message, const uint64_t client_id)
{
    impl->pushToQueue(message, client_id);
}
}    // namespace atcg
