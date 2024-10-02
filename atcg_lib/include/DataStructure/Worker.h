#pragma once

#include <Core/Memory.h>
#include <functional>

namespace atcg
{
class Worker
{
public:
    /**
     * @brief Job definition
     */
    using Job = std::function<void()>;

    /**
     * @brief Construct a new worker
     */
    Worker();

    /**
     * @brief Destructor
     */
    ~Worker();

    /**
     * @brief Push a job onto the job queue.
     * Jobs are done in the specified order of pushing onto this queue
     * @param job The new job
     */
    void pushJob(const Job& job);

    /**
     * @brief Start the worker.
     * This will run async to the calling thread. It directly starts to work on pending jobs or waits until new jobs are
     * pushed onto the queue.
     */
    void start();

    /**
     * @brief Start the worker.
     * Calling this function will block the calling thread until all jobs are done. However, the work is still performed
     * on a separate thread.
     */
    void startSync();

    /**
     * @brief If the worker is running.
     * Check if the worker is running. This function returns true if "start()" was called and the worker is either
     * performing jobs or waiting for new ones. To see if the worker completed all jobs, use isDone()
     *
     * @return If the worker was started
     */
    bool isRunning() const;

    /**
     * @brief Stop the worker.
     * This function stops the worker as soon as the current task is finished. Further tasks are considered and remain
     * in the queue.
     */
    void stop();

    /**
     * @brief Check if all jobs are done.
     * @note This function may also return true if stop was called but there are still pending jobs in the queue
     *
     * @return True if no jobs are in the queue and there are no jobs being worked on
     */
    bool isDone() const;

    /**
     * @brief Wait for all jobs to be done.
     * This function blocks the calling thread until the job queue is empty and no jobs are running
     */
    void waitDone() const;

    /**
     * @brief Clears all jobs from the queue
     */
    void clearQueue();

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg