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
    TCPClient();

    ~TCPClient();

    void connect(const std::string& ip, const uint32_t port);

    void disconnect();

    torch::Tensor sendAndWait(uint8_t* data, const uint32_t data_size);

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg