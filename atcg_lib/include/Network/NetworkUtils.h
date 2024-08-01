#pragma once

#include <Core/Platform.h>
#include <Math/Utils.h>

namespace atcg
{
namespace NetworkUtils
{

/**
 * @brief Read a byte from a data stream.
 *
 * @param data The data buffer
 * @param offset The read offset into the buffer. This value gets advanced by the number of read bytes
 * @return The requested data
 */
ATCG_INLINE uint8_t readByte(uint8_t* data, uint32_t& offset)
{
    uint8_t result = *(data + offset);
    offset += sizeof(uint8_t);
    return result;
}

/**
 * @brief Read an int from a data stream.
 * The data is assumed to be in network endianess (big) and is converted to the machine endianess.
 *
 * @tparam T The data type
 * @note Currently implemented for {u}int{16|32|64}_t.
 *
 * @param data The data buffer
 * @param offset The read offset into the buffer. This value gets advanced by the number of read bytes
 * @return The requested data
 */
template<typename T>
ATCG_INLINE T readInt(uint8_t* data, uint32_t& offset)
{
    T result = atcg::ntoh(*(T*)(data + offset));
    offset += sizeof(T);
    return result;
}

/**
 * @brief Read a string from a data stream.
 * It is assumed that the first 4 bytes at buffer[offset] are the length of the string, followed by an ascii
 * representation of the string where one character takes up one byte.
 *
 * @param data The data buffer
 * @param offset The read offset into the buffer. This value gets advanced by the number of read bytes
 * @return The requested data
 */
ATCG_INLINE std::string readString(uint8_t* data, uint32_t& offset)
{
    uint32_t stringlen = readInt<uint32_t>(data, offset);
    std::string result = std::string((char*)(data + offset), stringlen);
    offset += stringlen;
    return result;
}

/**
 * @brief Write a byte into a data buffer.
 *
 * @param data The data buffer to write to
 * @param offset The write offset. This value gets advanced by the number of written bytes
 * @param toWrite The data to write
 */
ATCG_INLINE void writeByte(uint8_t* data, uint32_t& offset, const uint8_t toWrite)
{
    *(uint8_t*)(data + offset) = toWrite;
    offset += sizeof(uint8_t);
}

/**
 * @brief Write an int into a data buffer.
 * This function converts the int from machine endianess to network endianess (big)
 *
 * @tparam T The data type
 * @note Currently implemented for {u}int{16|32|64}_t.
 *
 * @param data The data buffer to write to
 * @param offset The write offset. This value gets advanced by the number of written bytes
 * @param toWrite The data to write
 */
template<typename T>
ATCG_INLINE ATCG_INLINE void writeInt(uint8_t* data, uint32_t& offset, T toWrite)
{
    *(T*)(data + offset) = atcg::hton(toWrite);
    offset += sizeof(T);
}

/**
 * @brief Write a block of memory into a data buffer.
 * This function puts the size of the buffer into data[offset] converted to network endianess (big). The toWrite buffer
 * is then copied into the destination buffer without any conversions.
 *
 * @param data The data buffer to write to
 * @param offset The write offset. This value gets advanced by the number of written bytes
 * @param toWrite The data to write
 * @param size The size of the toWrite buffer
 */
ATCG_INLINE void writeBuffer(uint8_t* data, uint32_t& offset, uint8_t* toWrite, const uint32_t size)
{
    writeInt(data, offset, size);
    std::memcpy(data + offset, toWrite, size);
    offset += size;
}

/**
 * @brief Write a string to a data stream.
 * This function puts the size of the string into data[offset] converted to network endianess (big), followed by an
 * ascii representation of the string where one character takes up one byte.
 *
 * @param data The data buffer
 * @param offset The read offset into the buffer. This value gets advanced by the number of read bytes
 * @param toWrite The data to write
 */
ATCG_INLINE void writeString(uint8_t* data, uint32_t& offset, const std::string toWrite)
{
    writeBuffer(data, offset, (uint8_t*)toWrite.c_str(), toWrite.length());
}
}    // namespace NetworkUtils
}    // namespace atcg