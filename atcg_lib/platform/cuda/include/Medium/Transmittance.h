#pragma once

#include <DataStructure/Ray.h>
#include <Medium/Sampling.h>
#include <Utils/HostDevice.h>

namespace atcg
{
enum class TransmittanceSamplingStrategyType
{
    HOMOGENEOUS_TRACKING,
    DELTA_TRACKING,
    RATIO_TRACKING
};

template<TransmittanceSamplingStrategyType strategy, typename U = void>
struct TransmittanceEstimator;

template<>
struct TransmittanceEstimator<TransmittanceSamplingStrategyType::HOMOGENEOUS_TRACKING>
{
    float _density;
    ATCG_HOST_DEVICE ATCG_FORCE_INLINE TransmittanceEstimator(float density) : _density(density) {}

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE float estimate(const atcg::Ray& ray)
    {
        return glm::exp(-_density * (ray.tmax - ray.tmin));
    }
};

template<typename Grid>
struct TransmittanceEstimator<TransmittanceSamplingStrategyType::DELTA_TRACKING, Grid>
{
    float _max_density;
    Grid _density_grid;
    ATCG_HOST_DEVICE ATCG_FORCE_INLINE TransmittanceEstimator(float max_density, const Grid& density_grid)
        : _max_density(max_density),
          _density_grid(density_grid)
    {
    }

    ATCG_HOST_DEVICE float estimate(const atcg::Ray& ray, const glm::mat4& world_to_object, atcg::PCG32& rng)
    {
        float t = ray.tmin;
        SamplingStrategy<SamplingStrategyType::EXPONENTIAL_SAMPLING> sampling_strategy(_max_density);
        while(t < ray.tmax)
        {
            float sampled_distance = sampling_strategy.sample(rng.next1d());
            t += sampled_distance;
            if(t < ray.tmax)
            {
                glm::vec3 object_space_position =
                    atcg::Math::transformPoint(world_to_object, ray.origin + t * ray.direction);
                float density = _density_grid.eval(object_space_position);
                if(rng.next1d() < density / _max_density)
                {
                    return 0.0f;
                }
            }
        }
        return 1.0f;
    }
};

template<typename Grid>
struct TransmittanceEstimator<TransmittanceSamplingStrategyType::RATIO_TRACKING, Grid>
{
    float _majorant;
    Grid _density_grid;
    ATCG_HOST_DEVICE ATCG_FORCE_INLINE TransmittanceEstimator(float majorant, const Grid& density_grid)
        : _majorant(majorant),
          _density_grid(density_grid)
    {
    }

    ATCG_HOST_DEVICE float estimate(const atcg::Ray& ray, const glm::mat4& world_to_object, atcg::PCG32& rng)
    {
        float t             = ray.tmin;
        float transmittance = 1.0f;
        SamplingStrategy<SamplingStrategyType::EXPONENTIAL_SAMPLING> sampling_strategy(_majorant);
        while(t < ray.tmax)
        {
            float sampled_distance = sampling_strategy.sample(rng.next1d());
            t += sampled_distance;
            if(t < ray.tmax)
            {
                glm::vec3 object_space_position =
                    atcg::Math::transformPoint(world_to_object, ray.origin + t * ray.direction);
                float density = _density_grid.eval(object_space_position);
                transmittance *= glm::clamp(1.0f - density / _majorant, 0.0f, 1.0f);
            }
        }
        return transmittance;
    }
};
}    // namespace atcg