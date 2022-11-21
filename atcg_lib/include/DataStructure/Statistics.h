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
        Statistic(const std::string& name) :_name(name) {}

        /**
         * @brief Destructor 
         */
        ~Statistic() = default;

        /**
         * @brief Add a sample to the statistic 
         * 
         * @param sample The new sample
         */
        void addSample(const T& sample);

        /**
         * @brief Get the mean of the data points
         * 
         * @return Estimate of the mean
         */
        T mean() const;

        /**
         * @brief Get the (unbiased) variance of the data 
         * 
         * @return The variance
         */
        T var() const;

        /**
         * @brief Prints the mean and standard deviation of the underlying data 
         * 
         * @param os The ostream
         * @return The ostream
         */
        std::ostream& operator<<(std::ostream& os);
    private:
        std::vector<T> _samples;
        std::string _name;
    };

    template<typename T>
    void Statistic<T>::addSample(const T& sample)
    {
        _samples.push_back(sample);
    }

    template<typename T>
    T Statistic<T>::mean() const
    {
        T m = T(0);
        for(const T& sample : _samples)
        {
            m += sample;
        }

        return m/static_cast<T>(_samples.size());
    }

    template<typename T>
    T Statistic<T>::var() const
    {
        if(_samples.size() < 1) return T(0);

        T v = T(0);
        T m = mean();
        for(const T& sample : _samples)
        {
            m += (sample - m) * (sample - m);
        }

        return v/static_cast<T>(_samples.size() - 1);
    }

    template<typename T>
    std::ostream& Statistic<T>::operator<<(std::ostream& os)
    {
        os << "Statistic for " << _name << ":\t";
        os << mean() << "\t";
        os << "( " << std::sqrt(var()) << " )\n";
        return os;
    }
}