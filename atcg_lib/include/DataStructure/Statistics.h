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

template<typename T, class allocator>
class Collection : public MemoryBuffer<T, allocator>
{
public:
    Collection(const std::string& name) : MemoryBuffer<T, allocator>(), _name(name) {}

    Collection(const std::string& name, std::size_t n) : MemoryBuffer<T, allocator>(n), _name(name) {}

    ~Collection(){}

    ATCG_HOST_DEVICE
    virtual void add(const T& value);

    ATCG_HOST_DEVICE
    T mean() const;

    ATCG_HOST_DEVICE
    T var() const;

    ATCG_HOST_DEVICE
    void resetStatistics();

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
void Collection<T, allocator>::add(const T& value)
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
    _var   = 0;
    _M2    = 0;
    _index = 0;
    _count = 0;
}

template<typename T, class allocator>
class CyclicCollection : public Collection<T, allocator>
{
public:
    CyclicCollection(const std::string& name) : Collection<T, allocator>(name) {}

    CyclicCollection(const std::string& name, std::size_t n) : Collection<T, allocator>(name, n) {}

    ~CyclicCollection() {}

    ATCG_HOST_DEVICE
    virtual void add(const T& value) override;

    ATCG_HOST_DEVICE
    void resetStatistics();
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
void CyclicCollection<T, allocator>::add(const T& value)
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