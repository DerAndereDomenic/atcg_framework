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
    ATCG_HOST_DEVICE
    Statistic() = default;

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
     * @brief Remove a sample from the statistic.
     * The sample has to be added to the statistic prior to this call.
     *
     * @param sample The sample to remove
     */
    ATCG_HOST_DEVICE
    void removeSample(const T& sample);

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

    /**
     * @brief Get the number of elements in the collection
     *
     * @return The number of samples.
     */
    ATCG_HOST_DEVICE
    uint32_t count() const;

    /**
     * @brief Reset a statistic.
     */
    ATCG_HOST_DEVICE
    void reset();

private:
    std::string _name = "";
    T _mean           = T(0);
    T _M2             = T(0);
    uint32_t _count   = 0;
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
    ++_count;
    T delta = sample - _mean;
    _mean += delta / (T)_count;
    T delta2 = sample - _mean;
    _M2 += delta * delta2;
}

template<typename T>
void Statistic<T>::removeSample(const T& sample)
{
    --_count;
    T delta = sample - _mean;
    _mean -= delta / (T)(_count);
    T delta2 = sample - _mean;
    _M2 -= delta * delta2;
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

template<typename T>
uint32_t Statistic<T>::count() const
{
    return _count;
}

template<typename T>
void Statistic<T>::reset()
{
    _count = 0;
    _mean  = T(0);
    _M2    = T(0);
}

/**
 * @brief A class to model a collection that also holds a statistic about this collection
 *
 * @tparam T The type of the elements of the collection
 */
template<typename T>
class Collection : public MemoryBuffer<T, atcg::host_allocator>
{
public:
    /**
     * @brief Create a collection.
     *
     * @param name The name of the collection
     * @param n The capacity of the collection
     */
    Collection(const std::string& name, std::size_t n) : MemoryBuffer<T, atcg::host_allocator>(n), _name(name) {}

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
    virtual void addSample(const T& value);

    /**
     * @brief Get the mean of the collection.
     *
     * @return The mean
     */
    T mean() const;

    /**
     * @brief Get the variance of the collection.
     *
     * @return The variance
     */
    T var() const;

    /**
     * @brief Reset the statistic.
     * Also invalidates all the samples and clears the collection
     */
    void resetStatistics();

    /**
     * @brief Get the name of the collection
     *
     * @return The name
     */
    std::string name() const;

    /**
     * @brief Get the number of samples
     *
     * @return The number of samples
     */
    uint32_t count() const;

    /**
     * @brief Get the current index into the collection
     *
     * @return The current index
     */
    uint32_t index() const;

protected:
    uint32_t _index = 0;
    std::string _name;
    Statistic<T> _statistic;
};

/**
 * @brief Prints the mean and standard deviation of the underlying data
 *
 * @param os The ostream
 * @return The ostream
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, const Collection<T>& statistic)
{
    os << "Statistic for " << statistic.name() << ":\t";
    os << statistic.mean() << "\t";
    os << "( " << std::sqrt(statistic.var()) << " )\n";
    return os;
}

template<typename T>
void Collection<T>::addSample(const T& value)
{
    if(_index >= this->capacity())
    {
        ATCG_WARN("Collection is full");
        return;
    }

    this->get()[_index] = value;
    ++_index;
    _statistic.addSample(value);
}

template<typename T>
T Collection<T>::mean() const
{
    return _statistic.mean();
}

template<typename T>
T Collection<T>::var() const
{
    return _statistic.var();
}

template<typename T>
std::string Collection<T>::name() const
{
    return _name;
}

template<typename T>
void Collection<T>::resetStatistics()
{
    _statistic.reset();
    _index = 0;
}

template<typename T>
uint32_t Collection<T>::count() const
{
    return _statistic.count();
}

template<typename T>
uint32_t Collection<T>::index() const
{
    return _index;
}

/**
 * @brief A class to model a cyclic collection that also holds a statistic about this collection
 *
 * @tparam T The type of the elements of the collection
 */
template<typename T>
class CyclicCollection : public Collection<T>
{
public:
    /**
     * @brief Create a cyclic collection.
     *
     * @param name The name of the collection
     * @param n The capacity of the collection
     */
    CyclicCollection(const std::string& name, std::size_t n) : Collection<T>(name, n) {}

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
    virtual void addSample(const T& value) override;
};

/**
 * @brief Prints the mean and standard deviation of the underlying data
 *
 * @param os The ostream
 * @return The ostream
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, const CyclicCollection<T>& statistic)
{
    os << "Statistic for " << statistic.name() << ":\t";
    os << statistic.mean() << "\t";
    os << "( " << std::sqrt(statistic.var()) << " )\n";
    return os;
}

template<typename T>
void CyclicCollection<T>::addSample(const T& value)
{
    if(_statistic.count() >= this->capacity())
    {
        T value_old         = this->get()[_index];
        this->get()[_index] = value;

        // Remove old sample
        _statistic.removeSample(value_old);
    }
    else
    {
        this->get()[_index] = value;
    }

    _statistic.addSample(value);
    _index = (_index + 1) % this->capacity();
}


}    // namespace atcg