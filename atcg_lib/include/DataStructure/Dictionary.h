#pragma once

#include <Core/Memory.h>
#include <Core/Platform.h>

#include <unordered_map>
#include <any>

namespace atcg
{
/**
 * @brief A class to model a dictionary
 */
class Dictionary
{
public:
    Dictionary()  = default;
    ~Dictionary() = default;

    /**
     * @brief Set a value for a given key.
     * If the key already exists, the value will be overwritten.
     *
     * @tparam T The type of the value to store.
     * @param key The key for the value.
     * @param value The value to store. Will be forwarded (copied or moved).
     */
    template<typename T>
    ATCG_INLINE void setValue(std::string_view key, T&& value)
    {
        _values[std::string(key)] = std::forward<T>(value);
    }

    /**
     * @brief Retrieve a value of a given type for a key.
     * Throws std::out_of_range if the key is not found.
     * Throws std::bad_any_cast if the stored value type doesn't match T.
     *
     * @tparam T The expected type of the stored value.
     * @param key The key to look up.
     * @return The value associated with the key.
     */
    template<typename T>
    ATCG_INLINE T getValue(std::string_view key) const
    {
        auto it = _values.find(std::string(key));
        if(it == _values.end()) throw std::out_of_range("Key not found");

        return std::any_cast<T>(it->second);
    }

    /**
     * @brief Retrieve a value if it exists, or return a fallback value.
     * Returns the fallback if the key is missing or if the stored value cannot be cast to T.
     *
     * @tparam T The expected type of the stored value.
     * @param key The key to look up.
     * @param out The fallback value to return if lookup or cast fails.
     * @return The value from the map or the fallback.
     */
    template<typename T>
    ATCG_INLINE T getValueOr(std::string_view key, const T& out) const
    {
        auto it = _values.find(std::string(key));
        if(it == _values.end()) return out;

        if(auto val = std::any_cast<T>(&(it->second)))
        {
            return *val;
        }
        return out;
    }

    /**
     * @brief Remove a value associated with the key.
     * Does nothing if the key does not exist.
     *
     * @param key The key to remove.
     */
    ATCG_INLINE void remove(std::string_view key) { _values.erase(std::string(key)); }

    /**
     * @brief Check if a key exists in the dictionary.
     *
     * @param key The key to check.
     * @return true if the key exists, false otherwise.
     */
    ATCG_INLINE bool contains(std::string_view key) const { return _values.find(std::string(key)) != _values.end(); }

private:
    std::unordered_map<std::string, std::any> _values;
};
}    // namespace atcg