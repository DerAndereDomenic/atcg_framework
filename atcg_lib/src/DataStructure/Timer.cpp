#include <DataStructure/Timer.h>

namespace atcg
{
Timer::Timer()
{
    reset();
}

void Timer::reset()
{
    _start = std::chrono::high_resolution_clock::now();
}

float Timer::elapsedSeconds() const
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - _start)
               .count() *
           0.001f * 0.001f * 0.001f;
}

float Timer::elapsedMillis() const
{
    return elapsedSeconds() * 1000.0f;
}
}    // namespace atcg