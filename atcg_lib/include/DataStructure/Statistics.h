#pragma once

#include <iostream>
#include <vector>

namespace atcg
{
    template<typename T>
    class Statistic
    {
    public:
        Statistic(const std::string& name) :_name(name) {}

        ~Statistic() = default;

        void addSample(const T& sample);

        T mean() const;

        T var() const;

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
        os << "( " << var() << " )\n";
        return os;
    }
}