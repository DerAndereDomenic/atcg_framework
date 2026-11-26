#include "Bindings.h"

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
namespace
{
/**
 * @brief Check if a read of byte_size bytes at offset stays inside the buffer
 *
 * @param data The buffer
 * @param offset The read offset
 * @param byte_size The number of bytes to read
 * @param name The name of the calling function
 * @return True if the read is valid
 */
bool checkRead(const std::vector<uint8_t>& data, uint32_t offset, uint64_t byte_size, const char* name)
{
    if((uint64_t)offset + byte_size > (uint64_t)data.size())
    {
        ATCG_ERROR("Network.{0}: Reading {1} bytes at offset {2} exceeds the buffer size of {3} bytes",
                   name,
                   byte_size,
                   offset,
                   data.size());
        return false;
    }
    return true;
}

/**
 * @brief Make sure that byte_size bytes can be written at offset. The buffer is resized if needed.
 *
 * @param data The buffer
 * @param offset The write offset
 * @param byte_size The number of bytes to write
 */
void reserveWrite(std::vector<uint8_t>& data, uint32_t offset, uint64_t byte_size)
{
    uint64_t required_size = (uint64_t)offset + byte_size;
    if(required_size > (uint64_t)data.size())
    {
        data.resize(required_size);
    }
}

template<typename T>
std::pair<T, uint32_t> readIntChecked(std::vector<uint8_t>& data, uint32_t offset, const char* name)
{
    if(!checkRead(data, offset, sizeof(T), name))
    {
        return std::make_pair(T(0), offset);
    }

    T result = atcg::NetworkUtils::readInt<T>(data.data(), offset);
    return std::make_pair(result, offset);
}

template<typename T>
std::pair<std::vector<uint8_t>, uint32_t> writeIntChecked(std::vector<uint8_t>& data, uint32_t offset, T value)
{
    reserveWrite(data, offset, sizeof(T));
    atcg::NetworkUtils::writeInt<T>(data.data(), offset, value);
    return std::make_pair(data, offset);
}
}    // namespace

// Network submodule
void bindNetwork(BindingContext& ctx)
{
    py::module_ m = ctx.module();

    auto m_network    = m.def_submodule("Network");
    auto m_tcp_server = py::class_<atcg::TCPServer>(m_network, "TCPServer");
    auto m_tcp_client = py::class_<atcg::TCPClient>(m_network, "TCPClient");

    ctx.define(
        [=]() mutable
        {
            // The GIL is released for calls that block or wait for the network thread, which may call python callbacks
            m_tcp_server.def(py::init<>())
                .def("start", &atcg::TCPServer::start, "ip"_a, "port"_a, py::call_guard<py::gil_scoped_release>())
                .def("stop", &atcg::TCPServer::stop, py::call_guard<py::gil_scoped_release>())
                .def("sendToClient",
                     &atcg::TCPServer::sendToClient,
                     "message"_a,
                     "client_id"_a,
                     py::call_guard<py::gil_scoped_release>())
                .def("setOnConnectCallback", &atcg::TCPServer::setOnConnectCallback, "callback"_a)
                .def("setOnReceiveCallback", &atcg::TCPServer::setOnReceiveCallback, "callback"_a)
                .def("setOnDisconnectCallback", &atcg::TCPServer::setOnDisconnectCallback, "callback"_a);

            m_tcp_client.def(py::init<>())
                .def("connect", &atcg::TCPClient::connect, "ip"_a, "port"_a, py::call_guard<py::gil_scoped_release>())
                .def("disconnect", &atcg::TCPClient::disconnect, py::call_guard<py::gil_scoped_release>())
                .def("sendAndWait", &atcg::TCPClient::sendAndWait, "data"_a, py::call_guard<py::gil_scoped_release>());

            // Read functions return (value, new_offset). Out of range reads log an error and return a default value
            // with the unchanged offset.
            m_network.def(
                "readByte",
                [](std::vector<uint8_t>& data, uint32_t offset)
                {
                    if(!checkRead(data, offset, sizeof(uint8_t), "readByte"))
                    {
                        return std::make_pair(uint8_t(0), offset);
                    }
                    uint8_t result = atcg::NetworkUtils::readByte(data.data(), offset);
                    return std::make_pair(result, offset);
                },
                "data"_a,
                "offset"_a);

            m_network.def(
                "readInt16",
                [](std::vector<uint8_t>& data, uint32_t offset)
                { return readIntChecked<int16_t>(data, offset, "readInt16"); },
                "data"_a,
                "offset"_a);

            m_network.def(
                "readUInt16",
                [](std::vector<uint8_t>& data, uint32_t offset)
                { return readIntChecked<uint16_t>(data, offset, "readUInt16"); },
                "data"_a,
                "offset"_a);

            m_network.def(
                "readInt32",
                [](std::vector<uint8_t>& data, uint32_t offset)
                { return readIntChecked<int32_t>(data, offset, "readInt32"); },
                "data"_a,
                "offset"_a);

            m_network.def(
                "readUInt32",
                [](std::vector<uint8_t>& data, uint32_t offset)
                { return readIntChecked<uint32_t>(data, offset, "readUInt32"); },
                "data"_a,
                "offset"_a);

            m_network.def(
                "readInt64",
                [](std::vector<uint8_t>& data, uint32_t offset)
                { return readIntChecked<int64_t>(data, offset, "readInt64"); },
                "data"_a,
                "offset"_a);

            m_network.def(
                "readUInt64",
                [](std::vector<uint8_t>& data, uint32_t offset)
                { return readIntChecked<uint64_t>(data, offset, "readUInt64"); },
                "data"_a,
                "offset"_a);

            m_network.def(
                "readString",
                [](std::vector<uint8_t>& data, uint32_t offset)
                {
                    if(!checkRead(data, offset, sizeof(uint32_t), "readString"))
                    {
                        return std::make_pair(std::string(), offset);
                    }

                    // Peek the length of the string without moving the offset
                    uint32_t length_offset = offset;
                    uint32_t length        = atcg::NetworkUtils::readInt<uint32_t>(data.data(), length_offset);
                    if(!checkRead(data, length_offset, length, "readString"))
                    {
                        return std::make_pair(std::string(), offset);
                    }

                    std::string result = atcg::NetworkUtils::readString(data.data(), offset);
                    return std::make_pair(result, offset);
                },
                "data"_a,
                "offset"_a);

            // Write functions return (data, new_offset). The buffer is resized if it is too small.
            m_network.def(
                "writeByte",
                [](std::vector<uint8_t>& data, uint32_t offset, uint8_t value)
                {
                    reserveWrite(data, offset, sizeof(uint8_t));
                    atcg::NetworkUtils::writeByte(data.data(), offset, value);
                    return std::make_pair(data, offset);
                },
                "data"_a,
                "offset"_a,
                "value"_a);

            m_network.def(
                "writeInt16",
                [](std::vector<uint8_t>& data, uint32_t offset, int16_t value)
                { return writeIntChecked<int16_t>(data, offset, value); },
                "data"_a,
                "offset"_a,
                "value"_a);

            m_network.def(
                "writeUInt16",
                [](std::vector<uint8_t>& data, uint32_t offset, uint16_t value)
                { return writeIntChecked<uint16_t>(data, offset, value); },
                "data"_a,
                "offset"_a,
                "value"_a);

            m_network.def(
                "writeInt32",
                [](std::vector<uint8_t>& data, uint32_t offset, int32_t value)
                { return writeIntChecked<int32_t>(data, offset, value); },
                "data"_a,
                "offset"_a,
                "value"_a);

            m_network.def(
                "writeUInt32",
                [](std::vector<uint8_t>& data, uint32_t offset, uint32_t value)
                { return writeIntChecked<uint32_t>(data, offset, value); },
                "data"_a,
                "offset"_a,
                "value"_a);

            m_network.def(
                "writeInt64",
                [](std::vector<uint8_t>& data, uint32_t offset, int64_t value)
                { return writeIntChecked<int64_t>(data, offset, value); },
                "data"_a,
                "offset"_a,
                "value"_a);

            m_network.def(
                "writeUInt64",
                [](std::vector<uint8_t>& data, uint32_t offset, uint64_t value)
                { return writeIntChecked<uint64_t>(data, offset, value); },
                "data"_a,
                "offset"_a,
                "value"_a);

            m_network.def(
                "writeBuffer",
                [](std::vector<uint8_t>& data, uint32_t offset, std::vector<uint8_t>& value)
                {
                    reserveWrite(data, offset, sizeof(uint32_t) + value.size());
                    atcg::NetworkUtils::writeBuffer(data.data(), offset, value.data(), (uint32_t)value.size());
                    return std::make_pair(data, offset);
                },
                "data"_a,
                "offset"_a,
                "value"_a);

            m_network.def(
                "writeString",
                [](std::vector<uint8_t>& data, uint32_t offset, const std::string& value)
                {
                    reserveWrite(data, offset, sizeof(uint32_t) + value.length());
                    atcg::NetworkUtils::writeString(data.data(), offset, value);
                    return std::make_pair(data, offset);
                },
                "data"_a,
                "offset"_a,
                "value"_a);
        });
}
}    // namespace python
}    // namespace atcg
