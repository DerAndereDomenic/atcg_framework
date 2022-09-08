#pragma once

#include <chrono>

namespace atcg
{
    /**
     * @brief This class models a timer
     */
    class Timer
    {
    public:
        /**
         * @brief Construct a new Timer.
         * Creates and starts the timer
         */
        Timer();

        /**
         * @brief Reset the timer
         */
        void reset();

        /**
         * @brief Get the elapsed time in seconds
         * 
         * @return float The time
         */
        float elapsedSeconds() const;

        /**
         * @brief Get the elapsed time in milliseconds
         * 
         * @return float The time
         */
        float elapsedMillis() const;
    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> _start;
    };
};