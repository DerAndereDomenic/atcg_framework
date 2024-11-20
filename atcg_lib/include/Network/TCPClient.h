#pragma once

#include <SFML/Network.hpp>
#include <torch/types.h>

#include <memory>
#include <functional>

namespace atcg
{
/**
 * @brief Class to model a TCP client.
 */
class TCPClient
{
public:
    /**
     * @brief Create a simple TCP client.
     */
    TCPClient();

    /**
     * @brief Destructor
     */
    ~TCPClient();

    /**
     * @brief Connect to a server
     *
     * @param ip The server ip
     * @param port The server port
     */
    void connect(const std::string& ip, const uint32_t port);

    /**
     * @brief Disconnect drom the server
     */
    void disconnect();

    /**
     * @brief Send a buffer and wait for a response.
     *
     * @param data The data to send
     *
     * @return The response
     */
    std::vector<uint8_t> sendAndWait(const std::vector<uint8_t>& data);

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg