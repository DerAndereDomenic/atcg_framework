#pragma once

#include <Core/glm.h>
#include <Math/Color.h>

namespace atcg
{

template<uint32_t num_wavelength_samples>
struct SampledWavelengthsBase
{
    static ATCG_HOST_DEVICE SampledWavelengthsBase<num_wavelength_samples>
    sampleUniform(const float u, const float lambda_min, const float lambda_max)
    {
        SampledWavelengthsBase<num_wavelength_samples> result;
        result._wavelengths[0] = glm::mix(lambda_min, lambda_max, u);
        result._pdfs[0]        = 1.0f / (lambda_max - lambda_min);

        float delta = (lambda_max - lambda_min) / static_cast<float>(num_wavelength_samples);
        for(int i = 1; i < num_wavelength_samples; ++i)
        {
            result._wavelengths[i] = result._wavelengths[i - 1] + delta;
            if(result._wavelengths[i] > lambda_max)
            {
                result._wavelengths[i] = lambda_min + (result._wavelengths[i] - lambda_max);
            }
            result._pdfs[i] = 1.0f / (lambda_max - lambda_min);
        }
        return result;
    }

    static ATCG_HOST_DEVICE SampledWavelengthsBase<3> sampleRGB()
    {
        // Use center wavelengths of RGB channels
        SampledWavelengthsBase<3> result;
        result._wavelengths[0] = 620.0f;
        result._wavelengths[1] = 546.0f;
        result._wavelengths[2] = 455.0f;

        result._pdfs[0] = 1.0f;
        result._pdfs[1] = 1.0f;
        result._pdfs[2] = 1.0f;

        return result;
    }

    ATCG_INLINE ATCG_HOST_DEVICE float operator[](int i) const { return _wavelengths[i]; }

    ATCG_INLINE ATCG_HOST_DEVICE float& operator[](int i) { return _wavelengths[i]; }

    ATCG_INLINE ATCG_HOST_DEVICE float pdf(int i) const { return _pdfs[i]; }

    ATCG_INLINE ATCG_HOST_DEVICE bool secondaryTerminated() const
    {
        for(int i = 1; i < num_wavelength_samples; ++i)
        {
            if(_pdfs[i] > 0.0f) return false;
        }
        return true;
    }

    ATCG_INLINE ATCG_HOST_DEVICE void terminateSecondary()
    {
        for(int i = 1; i < num_wavelength_samples; ++i)
        {
            _pdfs[i] = 0.0f;
        }
        _pdfs[0] /= static_cast<float>(num_wavelength_samples);
    }

private:
    glm::vec<num_wavelength_samples, float> _wavelengths;
    glm::vec<num_wavelength_samples, float> _pdfs;
};

using SampledWavelengths1 = SampledWavelengthsBase<1>;
using SampledWavelengths2 = SampledWavelengthsBase<2>;
using SampledWavelengths3 = SampledWavelengthsBase<3>;
using SampledWavelengths4 = SampledWavelengthsBase<4>;

using SampledWavelengths = SampledWavelengths3;

template<int num_wavelength_samples>
struct SampledSpectrumBase : public glm::vec<num_wavelength_samples, float>
{
    ATCG_HOST_DEVICE SampledSpectrumBase() : glm::vec<num_wavelength_samples, float>(0.0f) {}

    ATCG_HOST_DEVICE SampledSpectrumBase(const glm::vec<num_wavelength_samples, float>& other)
        : glm::vec<num_wavelength_samples, float>(other)
    {
    }

    ATCG_HOST_DEVICE SampledSpectrumBase(const float& value) : glm::vec<num_wavelength_samples, float>(value) {}

    ATCG_HOST_DEVICE ATCG_INLINE bool hasNaNs() const
    {
        for(int i = 0; i < num_wavelength_samples; ++i)
        {
            if(glm::isnan((*this)[i])) return true;
        }
        return false;
    }

    ATCG_HOST_DEVICE ATCG_INLINE float average() const
    {
        float sum = 0.0f;
        for(int i = 0; i < num_wavelength_samples; ++i)
        {
            sum += (*this)[i];
        }
        return sum / static_cast<float>(num_wavelength_samples);
    }

    ATCG_HOST_DEVICE ATCG_INLINE float maxComponent() const
    {
        float max_val = (*this)[0];
        for(int i = 1; i < num_wavelength_samples; ++i)
        {
            max_val = glm::max(max_val, (*this)[i]);
        }
        return max_val;
    }

    ATCG_HOST_DEVICE ATCG_INLINE float minComponent() const
    {
        float min_val = (*this)[0];
        for(int i = 1; i < num_wavelength_samples; ++i)
        {
            min_val = glm::min(min_val, (*this)[i]);
        }
        return min_val;
    }

    ATCG_HOST_DEVICE ATCG_INLINE glm::vec3
    toXYZ(const SampledWavelengthsBase<num_wavelength_samples>& wavelengths) const
    {
        glm::vec3 xyz(0.0f);
        for(int i = 0; i < num_wavelength_samples; ++i)
        {
            float x_bar = Color::color_matching_x(wavelengths[i]);
            float y_bar = Color::color_matching_y(wavelengths[i]);
            float z_bar = Color::color_matching_z(wavelengths[i]);

            xyz.x += wavelengths.pdf(i) < 1e-4f ? 0.0f : (*this)[i] * x_bar / wavelengths.pdf(i);
            xyz.y += wavelengths.pdf(i) < 1e-4f ? 0.0f : (*this)[i] * y_bar / wavelengths.pdf(i);
            xyz.z += wavelengths.pdf(i) < 1e-4f ? 0.0f : (*this)[i] * z_bar / wavelengths.pdf(i);
        }
        return xyz / atcg::Constants::Y_integral<float>();
    }
};

using SampledSpectrum1 = SampledSpectrumBase<1>;
using SampledSpectrum2 = SampledSpectrumBase<2>;
using SampledSpectrum3 = SampledSpectrumBase<3>;
using SampledSpectrum4 = SampledSpectrumBase<4>;

using SampledSpectrum = SampledSpectrum3;
}    // namespace atcg