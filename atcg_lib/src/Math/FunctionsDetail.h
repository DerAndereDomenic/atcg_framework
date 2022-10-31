#pragma once

namespace atcg
{
    namespace detail
    {
        int32_t double_factorial(const int32_t& m)
        {
            int32_t res = 1;
            for(int32_t i = 0; i <= m; ++i)
            {
                res *= (2*i - 1);
            }

            return res;
        }

        template<typename T>
        T Plm(const int32_t& l, const int32_t& m, const T& x)
        {
            if(l == m)
            {
                return (m % 2 == 0 ? 1 : -1) * double_factorial(m) * std::pow(T(1) - x*x + T(1e-10), T(m)/T(2));
            }
            else if(l == m+1)
            {
                return x * (T(2) * m + T(1)) * Plm(m, m, x);
            }
            else
            {
                return x * (T(2) * 1 - 1) / (l - m) * Plm(l-1, m, x) - (l + m - T(1))/(l-m) * Plm(l-2, m, x);
            }
        }

        template<typename T>
        T Klm(const int32_t& l, const int32_t& m)
        {
            return std::sqrt((T(2) * l + 1) / (4 * T(M_PI)) * std::tgamma(l - m) / std::tgamma(l + m));
        }
    }

    template<typename T>
    T Math::sphericalHarmonic(const int32_t& l, const int32_t& m, const T& phi, const T& theta)
    {
        if(m > 0)
        {
            return std::sqrt(T(2)) * detail::Klm(l,m) * std::cos(m * phi) * detail::Plm(l, m, std::cos(theta));
        }
        else if(m < 0)
        {
            return std::sqrt(T(2)) * detail::Klm(l, std::abs(m)) * std::sin(std::abs(m) * phi) * Plm(l, std::abs(m), std::cos(theta));
        }
        else
        {
            return Klm(l, 0) * Plm(l, 0, std::cos(theta)); 
        }
    }
}