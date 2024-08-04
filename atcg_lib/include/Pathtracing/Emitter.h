#pragma once

namespace atcg
{
class Emitter
{
public:
    Emitter() = default;

    virtual ~Emitter() {}
};

struct EmitterSamplingResult
{
    glm::vec3 direction_to_light;
    float distance_to_light;
    glm::vec3 normal_at_light;
    glm::vec3 radiance_weight_at_receiver;
    float sampling_pdf;
};

struct PhotonSamplingResult
{
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 normal;
    glm::vec3 radiance_weight;    // Le / p in area measure
    float pdf;                    // 1/Area
};
}    // namespace atcg