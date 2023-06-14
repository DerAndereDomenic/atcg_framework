#include <Core/UUID.h>

#include <random>

namespace atcg
{
static std::random_device s_random_device;
static std::mt19937_64 s_engine(s_random_device());
static std::uniform_int_distribution<uint64_t> s_uniform_distribution;

UUID::UUID() : _UUID(s_uniform_distribution(s_engine)) {}

UUID::UUID(uint64_t uuid) : _UUID(uuid) {}
}    // namespace atcg