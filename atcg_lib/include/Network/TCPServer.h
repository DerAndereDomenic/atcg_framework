#pragma once

#include <SFML/Network.hpp>

#include <memory>
#include <functional>

namespace atcg
{

/**
 * @brief Class to model a TCP server.
 * When calling server->start(); it starts a new thread that listens to newly incomming directions and calls a user
 * defined callback when receiving data.
 */
class TCPServer
{
public:
    /**
     * @brief A callback function to act on received data.
     *
     * @param data The data to send
     * @param data_size The size of the buffer in size
     * @param client_id The id of the client
     */
    using NetworkCallbackFn = std::function<void(std::vector<uint8_t>&, const uint64_t)>;

    /**
     * @brief A callback funciton to act on connections
     *
     * @param client_id The id of the client
     */
    using NetworkConnectionCallbackFn = std::function<void(const uint64_t)>;

    /**
     * @brief Default constructor
     */
    TCPServer();

    /**
     * @brief Destructor
     */
    ~TCPServer();

    /**
     * @brief Start the server.
     * This starts a separate network thread
     *
     * @param ip The ip
     * @param port The port
     */
    void start(const std::string& ip, const uint32_t port);

    /**
     * @brief Stop the server.
     */
    void stop();

    /**
     * @brief Set the callback that is called when a new client connection is established
     *
     * @param callback The callback
     */
    void setOnConnectCallback(const NetworkConnectionCallbackFn& callback);

    /**
     * @brief Set the callback that is called when a client receives data
     *
     * @param callback The callback
     */
    void setOnReceiveCallback(const NetworkCallbackFn& callback);

    /**
     * @brief Set the callback that is called when a client disconnects
     *
     * @param callback The callback
     */
    void setOnDisconnectCallback(const NetworkConnectionCallbackFn& callback);

    /**
     * @brief Send data to a client
     *
     * @param message The data to send
     * @param client_id The id of the client to send to
     */
    void sendToClient(const std::vector<uint8_t>& message, const uint64_t client_id);

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg
