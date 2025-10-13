#pragma once

#include <Core/CUDA.h>

namespace atcg
{
// Pseudorandom number generation using the Tiny Encryption Algorithm (TEA) by David Wheeler and Roger Needham.
static inline ATCG_HOST_DEVICE uint32_t sampleTEA32(uint32_t val0, uint32_t val1, uint32_t rounds = 4)
{
    uint32_t v0 = val0;
    uint32_t v1 = val1;
    uint32_t s0 = 0;

    for(uint32_t n = 0; n < rounds; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

// Pseudorandom number generation using the Tiny Encryption Algorithm (TEA) by David Wheeler and Roger Needham.
static inline ATCG_HOST_DEVICE uint64_t sampleTEA64(uint32_t val0, uint32_t val1, uint32_t rounds = 4)
{
    uint32_t v0 = val0;
    uint32_t v1 = val1;
    uint32_t s0 = 0;

    for(uint32_t n = 0; n < rounds; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return static_cast<uint64_t>(v0) | (static_cast<uint64_t>(v1) << 32);
}


constexpr uint64_t PCG32_DEFAULT_STATE  = 0x853c49e6748fea9bULL;
constexpr uint64_t PCG32_DEFAULT_STREAM = 0xda3e39cb94b95bdbULL;
constexpr uint64_t PCG32_MULT           = 0x5851f42d4c957f2dULL;

/// PCG32 pseudorandom number generator proposed by Melissa O'Neill
struct PCG32
{
    inline ATCG_HOST_DEVICE PCG32(uint64_t initstate = PCG32_DEFAULT_STATE, uint64_t initseq = PCG32_DEFAULT_STREAM)
    {
        seed(initstate, initseq);
    }

    inline ATCG_HOST_DEVICE void seed(uint64_t initstate, uint64_t initseq)
    {
        state = 0;
        inc   = (initseq << 1) | 1;
        nextUint32();
        state += initstate;
        nextUint32();
    }

    inline ATCG_HOST_DEVICE uint32_t nextUint32()
    {
        uint64_t oldstate   = state;
        state               = oldstate * PCG32_MULT + inc;
        uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18) ^ oldstate) >> 27);
        uint32_t rot_offset = static_cast<uint32_t>(oldstate >> 59);
        return (xorshifted >> rot_offset) | (xorshifted << (32 - rot_offset));
    }

    inline ATCG_HOST_DEVICE uint64_t nextUint64()
    {
        return static_cast<uint64_t>(nextUint32()) | (static_cast<uint64_t>(nextUint32()) << 32);
    }

    // Generate a random float in [0, 1) containing 23 bits of randomness
    inline ATCG_HOST_DEVICE float nextFloat()
    {
        // First generate a random number in [1, 2) and subtract 1.
        uint32_t bits = (nextUint32() >> 9) | 0x3f800000u;
        return reinterpret_cast<float&>(bits) - 1.0f;
    }

    // Generate a random double in [0, 1) containing 32 bits of randomness (lower mantissa bits are set to 0 here)
    inline ATCG_HOST_DEVICE double nextDouble()
    {
        // First generate a random number in [1, 2) and subtract 1
        uint64_t bits = (static_cast<uint64_t>(nextUint32()) << 20) | 0x3ff0000000000000ull;
        return reinterpret_cast<double&>(bits) - 1.0;
    }


    // Generate a 1d sample in [0, 1)
    inline ATCG_HOST_DEVICE float next1d() { return nextFloat(); }

    // Generate a 2d sample in [0, 1)^2
    inline ATCG_HOST_DEVICE glm::vec2 next2d() { return glm::vec2(nextFloat(), nextFloat()); }

    // Generate a 3d sample in [0, 1)^3
    inline ATCG_HOST_DEVICE glm::vec3 next3d() { return glm::vec3(nextFloat(), nextFloat(), nextFloat()); }

    // Generate a 4d sample in [0, 1)^4
    inline ATCG_HOST_DEVICE glm::vec4 next4d() { return glm::vec4(nextFloat(), nextFloat(), nextFloat(), nextFloat()); }

    // Generate a 4d normal distributed sample
    inline ATCG_HOST_DEVICE glm::vec4 normal4d()
    {
        glm::vec2 u1 = 1.0f - next2d();
        glm::vec2 u2 = 1.0f - next2d();

        glm::vec2 r = glm::sqrt(-2.0f * glm::log(glm::max(glm::vec2(1e-10f), u1)));
        glm::vec2 t = 2.0f * glm::pi<float>() * u2;

        return glm::vec4(r.x * glm::cos(t.x), r.x * glm::sin(t.x), r.y * glm::cos(t.y), r.y * glm::sin(t.y));
    }

    // Generate a 3d normal distributed sample
    inline ATCG_HOST_DEVICE glm::vec3 normal3d() { return normal4d(); }

    // Generate a 2d normal distributed sample
    inline ATCG_HOST_DEVICE glm::vec2 normal2d()
    {
        float u1 = 1.0f - next1d();
        float u2 = 1.0f - next1d();
        float r  = glm::sqrt(-2.0f * glm::log(glm::max(1e-10f, u1)));
        float t  = 2.0f * glm::pi<float>() * u2;

        return r * glm::vec2(glm::cos(t), glm::sin(t));
    }

    // Generate a 1d normal distributed sample
    inline ATCG_HOST_DEVICE float normal1d() { return normal2d().x; }


    uint64_t state;    // RNG state.  All values are possible.
    uint64_t inc;      // Controls which RNG sequence (stream) is selected. Must *always* be odd.
};
}    // namespace atcg