#pragma once

namespace atcg
{
/**
 * @brief Select the backend for JPEG Encoder and Decoder
 */
enum class JPEGBackend
{
    SOFTWARE = 0,
    HARDWARE = 1
};
}    // namespace atcg