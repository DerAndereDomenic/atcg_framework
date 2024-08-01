#pragma once

#include <Math/Utils.h>

namespace atcg
{
namespace NetworkUtils
{
uint8_t readByte(uint8_t* data, uint32_t& offset)
{
    uint8_t result = *(data + offset);
    offset += sizeof(uint8_t);
    return result;
}

template<typename T>
T readInt(uint8_t* data, uint32_t& offset)
{
    T result = atcg::ntoh(*(T*)(data + offset));
    offset += sizeof(T);
    return result;
}

std::string readString(uint8_t* data, uint32_t& offset)
{
    uint32_t stringlen = readInt<uint32_t>(data, offset);
    std::string result = std::string((char*)(data + offset), stringlen);
    offset += stringlen;
    return result;
}

void writeByte(uint8_t* data, uint32_t& offset, const uint8_t toWrite)
{
    *(uint8_t*)(data + offset) = toWrite;
    offset += sizeof(uint8_t);
}

template<typename T>
void writeInt(uint8_t* data, uint32_t& offset, T toWrite)
{
    *(T*)(data + offset) = atcg::hton(toWrite);
    offset += sizeof(T);
}

void writeBuffer(uint8_t* data, uint32_t& offset, uint8_t* toWrite, const uint32_t size)
{
    writeInt(data, offset, size);
    std::memcpy(data + offset, toWrite, size);
    offset += size;
}

void writeString(uint8_t* data, uint32_t& offset, const std::string toWrite)
{
    writeBuffer(data, offset, (uint8_t*)toWrite.c_str(), toWrite.length());
}
}    // namespace NetworkUtils
}    // namespace atcg