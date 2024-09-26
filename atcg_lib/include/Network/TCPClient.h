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
     * @note It is expected that the first 4 bytes of data is the size of the buffer in big endian.
     * For example, if we want to send 16 bytes to the server, the actual amount of memory allocated in data should be
     * at least 20 bytes where atcg::NetworkUtils::readInt<uint32_t>(data, 0) == 16
     *
     * @param data The data to send
     *
     * @return The response
     */
    torch::Tensor sendAndWait(uint8_t* data);

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg