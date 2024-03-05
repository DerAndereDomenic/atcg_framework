#pragma once

#include <iostream>
#include <vector>

namespace atcg
{
/**
 * @brief A class to model a statistic
 * @tparam The type of data
 */
template<typename T>
class Statistic
{
public:
    /**
     * @brief Create a new statistic.
     *
     * @param name The name of the statistic
     */
    ATCG_HOST_DEVICE
    Statistic(const std::string& name) : _name(name) {}

    /**
     * @brief Destructor
     */
    ~Statistic() = default;

    /**
     * @brief Add a sample to the statistic
     *
     * @param sample The new sample
     */
    ATCG_HOST_DEVICE
    void addSample(const T& sample);

    /**
     * @brief Get the mean of the data points
     *
     * @return Estimate of the mean
     */
    ATCG_HOST_DEVICE
    T mean() const;

    /**
     * @brief Get the (unbiased) variance of the data
     *
     * @return The variance
     */
    ATCG_HOST_DEVICE
    T var() const;

    /**
     * @brief Get the name
     *
     * @return The name
     */
    ATCG_HOST_DEVICE
    std::string name() const;

private:
    std::string _name;
    T _mean         = 0;
    T _var          = 0;
    uint32_t _count = 0;
    T _M2           = 0;
};

/**
 * @brief Prints the mean and standard deviation of the underlying data
 *
 * @param os The ostream
 * @return The ostream
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, const Statistic<T>& statistic)
{
    os << "Statistic for " << statistic.name() << ":\t";
    os << statistic.mean() << "\t";
    os << "( " << std::sqrt(statistic.var()) << " )\n";
    return os;
}

template<typename T>
void Statistic<T>::addSample(const T& sample)
{
    T delta = sample - _mean;
    _mean += delta / (T)_count;
    T delta2 = sample - _mean;
    _M2 += delta * delta2;
}

template<typename T>
T Statistic<T>::mean() const
{
    return _mean;
}

template<typename T>
T Statistic<T>::var() const
{
    return _M2 / (T)_count;
}

template<typename T>
std::string Statistic<T>::name() const
{
    return _name;
}

/**
 * @brief A class to model a collection that also holds a statistic about this collection
 *
 * @tparam T The type of the elements of the collection
 * @tparam allocator Host or Device allocator
 */
template<typename T, class allocator>
class Collection : public MemoryBuffer<T, allocator>
{
public:
    /**
     * @brief Create a collection.
     *
     * @param name The name of the collection
     * @param n The capacity of the collection
     */
    Collection(const std::string& name, std::size_t n) : MemoryBuffer<T, allocator>(n), _name(name) {}

    /**
     * @brief The destructor
     */
    ~Collection() {}

    /**
     * @brief Add a sample.
     * If the capacity of the collection is reached, a warning is thrown and no elements are updated
     *
     * @param value The new sample
     */
    ATCG_HOST_DEVICE
    virtual void addSample(const T& value);

    /**
     * @brief Get the mean of the collection.
     *
     * @return The mean
     */
    ATCG_HOST_DEVICE
    T mean() const;

    /**
     * @brief Get the variance of the collection.
     *
     * @return The variance
     */
    ATCG_HOST_DEVICE
    T var() const;

    /**
     * @brief Reset the statistic.
     * Also invalidates all the samples and clears the collection
     */
    ATCG_HOST_DEVICE
    void resetStatistics();

    /**
     * @brief Get the name of the collection
     *
     * @return The name
     */
    ATCG_HOST_DEVICE
    std::string name() const;

protected:
    uint32_t _index = 0;
    uint32_t _count = 0;
    T _mean         = 0;
    T _var          = 0;
    T _M2           = 0;
    std::string _name;
};

/**
 * @brief Prints the mean and standard deviation of the underlying data
 *
 * @param os The ostream
 * @return The ostream
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, const Collection<T, atcg::host_allocator>& statistic)
{
    os << "Statistic for " << statistic.name() << ":\t";
    os << statistic.mean() << "\t";
    os << "( " << std::sqrt(statistic.var()) << " )\n";
    return os;
}

template<typename T, class allocator>
void Collection<T, allocator>::addSample(const T& value)
{
    if(_index >= capacity())
    {
        ATCG_WARN("Collection is full");
        return;
    }

    get()[_index] = value;
    ++_index;
    ++_count;

    T delta = value - _mean;
    _mean += delta / (T)_count;
    T delta2 = value - _mean;
    _M2 += delta * delta2;
}

template<typename T, class allocator>
T Collection<T, allocator>::mean() const
{
    return _mean;
}

template<typename T, class allocator>
T Collection<T, allocator>::var() const
{
    return _M2 / (T)_count;
}

template<typename T, class allocator>
std::string Collection<T, allocator>::name() const
{
    return _name;
}

template<typename T, class allocator>
void Collection<T, allocator>::resetStatistics()
{
    _mean  = 0;
    _M2    = 0;
    _index = 0;
    _count = 0;
}

/**
 * @brief A class to model a cyclic collection that also holds a statistic about this collection
 *
 * @tparam T The type of the elements of the collection
 * @tparam allocator Host or Device allocator
 */
template<typename T, class allocator>
class CyclicCollection : public Collection<T, allocator>
{
public:
    /**
     * @brief Create a cyclic collection.
     *
     * @param name The name of the collection
     * @param n The capacity of the collection
     */
    CyclicCollection(const std::string& name, std::size_t n) : Collection<T, allocator>(name, n) {}

    /**
     * @brief The destructor
     */
    ~CyclicCollection() {}

    /**
     * @brief Add a sample.
     * If the capacity of the collection is reached, the oldest element will be overwritten.
     *
     * @param value The new sample
     */
    ATCG_HOST_DEVICE
    virtual void addSample(const T& value) override;
};

/**
 * @brief Prints the mean and standard deviation of the underlying data
 *
 * @param os The ostream
 * @return The ostream
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, const CyclicCollection<T, atcg::host_allocator>& statistic)
{
    os << "Statistic for " << statistic.name() << ":\t";
    os << statistic.mean() << "\t";
    os << "( " << std::sqrt(statistic.var()) << " )\n";
    return os;
}

template<typename T, class allocator>
void CyclicCollection<T, allocator>::addSample(const T& value)
{
    if(_count >= capacity())
    {
        T value_old   = get()[_index];
        get()[_index] = value;

        // Remove old sample
        T delta = value_old - _mean;
        _mean -= delta / (T)(_count - 1);
        T delta2 = value_old - _mean;
        _M2 -= delta * delta2;
    }
    else
    {
        get()[_index] = value;
        ++_count;
    }

    T delta = value - _mean;
    _mean += delta / (T)_count;
    T delta2 = value - _mean;
    _M2 += delta * delta2;
    _index = (_index + 1) % capacity();
}


}    // namespace atcg