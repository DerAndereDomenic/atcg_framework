#pragma once

#include <DataStructure/Worker.h>

namespace atcg
{
class WorkerPool
{
public:
    /**
     * @brief Construct a worker pool for asynchronous jobs.
     *
     * @param num_workers The number of workers in the pool
     */
    WorkerPool(uint32_t num_workers);

    /**
     * @brief Destructor
     */
    ~WorkerPool();

    /**
     * @brief Push a job onto the job queue.
     * This function chooses a free worker from the pool arbitrarily. If all workers have pending jobs, the worker with
     * the smallest amount of pending jobs is chosen.
     * @note Note that "endless" jobs may block the queue and jobs that are pushed on the same worker will never be
     * processed.
     *
     * @param job The new job
     */
    void pushJob(const Worker::Job& job);

    /**
     * @brief Start the worker pool.
     * This will run async to the calling thread. It directly starts to work on pending jobs or waits until new jobs are
     * pushed onto the queue.
     */
    void start();

    /**
     * @brief Start the worker pool.
     * Calling this function will block the calling thread until all jobs are done. However, the work is still performed
     * on a separate threads. The jobs are still done in an asychronous way. Only the calling thread waits until all
     * work is completed.
     */
    void startSync();

    /**
     * @brief If the pool is running.
     * Check if the pool is running. This function returns true if "start()" was called and any worker is either
     * performing jobs or waiting for new ones. To see if all workers completed all jobs, use isDone()
     *
     * @return If the pool was started
     */
    bool isRunning() const;

    /**
     * @brief Stop the pool.
     * This function stops all workers as soon as the current task for each worker is finished is finished. Further
     * tasks are considered and remain in the queue.
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

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg