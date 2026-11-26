#include "Bindings.h"

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
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
            m_tcp_server.def(py::init<>())
                .def("start", &atcg::TCPServer::start)
                .def("stop", &atcg::TCPServer::stop)
                .def("sendToClient", &atcg::TCPServer::sendToClient)
                .def("setOnConnectCallback", &atcg::TCPServer::setOnConnectCallback)
                .def("setOnReceiveCallback", &atcg::TCPServer::setOnReceiveCallback)
                .def("setOnDisconnectCallback", &atcg::TCPServer::setOnDisconnectCallback);

            m_tcp_client.def(py::init<>())
                .def("connect", &atcg::TCPClient::connect)
                .def("disconnect", &atcg::TCPClient::disconnect)
                .def("sendAndWait", &atcg::TCPClient::sendAndWait);

            m_network.def("readByte",
                          [](std::vector<uint8_t>& data, uint32_t& offset)
                          {
                              uint8_t result = atcg::NetworkUtils::readByte(data.data(), offset);
                              return std::make_pair(result, offset);
                          });

            m_network.def("readInt16",
                          [](std::vector<uint8_t>& data, uint32_t& offset)
                          {
                              int16_t result = atcg::NetworkUtils::readInt<int16_t>(data.data(), offset);
                              return std::make_pair(result, offset);
                          });

            m_network.def("readUInt16",
                          [](std::vector<uint8_t>& data, uint32_t& offset)
                          {
                              uint16_t result = atcg::NetworkUtils::readInt<uint16_t>(data.data(), offset);
                              return std::make_pair(result, offset);
                          });

            m_network.def("readInt32",
                          [](std::vector<uint8_t>& data, uint32_t& offset)
                          {
                              int32_t result = atcg::NetworkUtils::readInt<int32_t>(data.data(), offset);
                              return std::make_pair(result, offset);
                          });

            m_network.def("readUInt32",
                          [](std::vector<uint8_t>& data, uint32_t& offset)
                          {
                              uint32_t result = atcg::NetworkUtils::readInt<uint32_t>(data.data(), offset);
                              return std::make_pair(result, offset);
                          });

            m_network.def("readInt64",
                          [](std::vector<uint8_t>& data, uint32_t& offset)
                          {
                              int64_t result = atcg::NetworkUtils::readInt<int64_t>(data.data(), offset);
                              return std::make_pair(result, offset);
                          });

            m_network.def("readUInt64",
                          [](std::vector<uint8_t>& data, uint32_t& offset)
                          {
                              uint64_t result = atcg::NetworkUtils::readInt<uint64_t>(data.data(), offset);
                              return std::make_pair(result, offset);
                          });

            m_network.def("readString",
                          [](std::vector<uint8_t>& data, uint32_t& offset)
                          {
                              auto result = atcg::NetworkUtils::readString(data.data(), offset);
                              return std::make_pair(result, offset);
                          });

            m_network.def("writeByte",
                          [](std::vector<uint8_t>& data, uint32_t& offset, uint8_t toWrite)
                          {
                              atcg::NetworkUtils::writeByte(data.data(), offset, toWrite);
                              return std::make_pair(data, offset);
                          });

            m_network.def("writeInt16",
                          [](std::vector<uint8_t>& data, uint32_t& offset, int16_t toWrite)
                          {
                              atcg::NetworkUtils::writeInt(data.data(), offset, toWrite);
                              return std::make_pair(data, offset);
                          });

            m_network.def("writeUInt16",
                          [](std::vector<uint8_t>& data, uint32_t& offset, uint16_t toWrite)
                          {
                              atcg::NetworkUtils::writeInt(data.data(), offset, toWrite);
                              return std::make_pair(data, offset);
                          });

            m_network.def("writeInt32",
                          [](std::vector<uint8_t>& data, uint32_t& offset, int32_t toWrite)
                          {
                              atcg::NetworkUtils::writeInt(data.data(), offset, toWrite);
                              return std::make_pair(data, offset);
                          });

            m_network.def("writeUInt32",
                          [](std::vector<uint8_t>& data, uint32_t& offset, uint32_t toWrite)
                          {
                              atcg::NetworkUtils::writeInt(data.data(), offset, toWrite);
                              return std::make_pair(data, offset);
                          });

            m_network.def("writeInt64",
                          [](std::vector<uint8_t>& data, uint32_t& offset, int64_t toWrite)
                          {
                              atcg::NetworkUtils::writeInt(data.data(), offset, toWrite);
                              return std::make_pair(data, offset);
                          });

            m_network.def("writeUInt64",
                          [](std::vector<uint8_t>& data, uint32_t& offset, uint64_t toWrite)
                          {
                              atcg::NetworkUtils::writeInt(data.data(), offset, toWrite);
                              return std::make_pair(data, offset);
                          });

            m_network.def("writeBuffer",
                          [](std::vector<uint8_t>& data, uint32_t& offset, std::vector<uint8_t>& toWrite)
                          {
                              atcg::NetworkUtils::writeBuffer(data.data(), offset, toWrite.data(), toWrite.size());
                              return std::make_pair(data, offset);
                          });

            m_network.def("writeString",
                          [](std::vector<uint8_t>& data, uint32_t& offset, const std::string& toWrite)
                          {
                              atcg::NetworkUtils::writeString(data.data(), offset, toWrite);
                              return std::make_pair(data, offset);
                          });
        });
}
}    // namespace python
}    // namespace atcg
