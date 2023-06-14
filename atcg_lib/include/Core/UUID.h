#pragma once

namespace atcg
{
/**
 * @brief A class to model an UUID
 */
class UUID
{
public:
    /**
     * @brief Constructor
     */
    UUID();

    /**
     * @brief Constructor from uint64_t
     *
     * @param uuid The uuid
     */
    UUID(uint64_t uuid);

    /**
     * @brief Copy constructor
     *
     */
    UUID(const UUID&) = default;

    /**
     * @brief Get the uuid as uint64_t
     *
     * @return The UUID
     */
    operator uint64_t() const { return _UUID; }

private:
    uint64_t _UUID;
}
}    // namespace atcg