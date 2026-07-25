#pragma once

#include <Core/Memory.h>
#include <Core/Platform.h>

#include <torch/types.h>

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
    Dictionary()                                 = default;
    ~Dictionary()                                = default;
    Dictionary(Dictionary&&) noexcept            = default;
    Dictionary& operator=(Dictionary&&) noexcept = default;

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
        using U                   = std::decay_t<T>;    // strip references and cv-qualifiers
        _values[std::string(key)] = std::any(U(std::forward<T>(value)));
    }

    /**
     * @brief Retrieve a value of a given type for a key as any object
     *
     * @param key The key to look up.
     * @return The value associated with the key.
     */
    ATCG_INLINE std::any getValueRaw(std::string_view key) const
    {
        auto it = _values.find(std::string(key));
        if(it == _values.end()) throw std::out_of_range("Key not found");

        return it->second;
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

    // Convenience functions for common types
    ATCG_INLINE void setInt8(std::string_view key, int8_t value) { setValue(key, value); }
    ATCG_INLINE void setInt16(std::string_view key, int16_t value) { setValue(key, value); }
    ATCG_INLINE void setInt32(std::string_view key, int32_t value) { setValue(key, value); }
    ATCG_INLINE void setInt64(std::string_view key, int64_t value) { setValue(key, value); }
    ATCG_INLINE void setUInt8(std::string_view key, uint8_t value) { setValue(key, value); }
    ATCG_INLINE void setUInt16(std::string_view key, uint16_t value) { setValue(key, value); }
    ATCG_INLINE void setUInt32(std::string_view key, uint32_t value) { setValue(key, value); }
    ATCG_INLINE void setUInt64(std::string_view key, uint64_t value) { setValue(key, value); }
    ATCG_INLINE void setFloat(std::string_view key, float value) { setValue(key, value); }
    ATCG_INLINE void setDouble(std::string_view key, double value) { setValue(key, value); }
    ATCG_INLINE void setString(std::string_view key, const std::string& value) { setValue(key, value); }
    ATCG_INLINE void setTensor(std::string_view key, torch::Tensor value) { setValue(key, value); }
    template<typename T>
    ATCG_INLINE void setPointer(std::string_view key, const atcg::ref_ptr<T>& value)
    {
        setValue(key, value);
    }

    ATCG_INLINE int8_t getInt8(std::string_view key) const { return getValue<int8_t>(key); }
    ATCG_INLINE int16_t getInt16(std::string_view key) const { return getValue<int16_t>(key); }
    ATCG_INLINE int32_t getInt32(std::string_view key) const { return getValue<int32_t>(key); }
    ATCG_INLINE int64_t getInt64(std::string_view key) const { return getValue<int64_t>(key); }
    ATCG_INLINE uint8_t getUInt8(std::string_view key) const { return getValue<uint8_t>(key); }
    ATCG_INLINE uint16_t getUInt16(std::string_view key) const { return getValue<uint16_t>(key); }
    ATCG_INLINE uint32_t getUInt32(std::string_view key) const { return getValue<uint32_t>(key); }
    ATCG_INLINE uint64_t getUInt64(std::string_view key) const { return getValue<uint64_t>(key); }
    ATCG_INLINE float getFloat(std::string_view key) const { return getValue<float>(key); }
    ATCG_INLINE double getDouble(std::string_view key) const { return getValue<double>(key); }
    ATCG_INLINE std::string getString(std::string_view key) const { return getValue<std::string>(key); }
    ATCG_INLINE torch::Tensor getTensor(std::string_view key) const { return getValue<torch::Tensor>(key); }
    template<typename T>
    ATCG_INLINE atcg::ref_ptr<T> getPointer(std::string_view key) const
    {
        return getValue<atcg::ref_ptr<T>>(key);
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