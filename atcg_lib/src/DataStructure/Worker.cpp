#include <DataStructure/Worker.h>

#include <queue>
#include <thread>
#include <mutex>
#include <chrono>

namespace atcg
{
class Worker::Impl
{
public:
    Impl();
    ~Impl();

    void worker_loop();

    std::queue<Job> jobs;
    std::mutex queue_mutex;
    std::atomic_bool running     = false;
    std::atomic_bool job_running = false;
    std::thread worker_thread;
};

Worker::Impl::Impl() {}

Worker::Impl::~Impl() {}

void Worker::Impl::worker_loop()
{
    while(running)
    {
        Job current_job = nullptr;
        {
            std::lock_guard guard(queue_mutex);
            if(!jobs.empty())
            {
                job_running = true;
                current_job = jobs.front();
                jobs.pop();
            }
        }

        if(current_job)
        {
            current_job();
        }
        else
        {
            job_running = false;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    job_running = false;
}

Worker::Worker()
{
    impl = std::make_unique<Impl>();
}

Worker::~Worker()
{
    if(impl->running)
    {
        waitDone();
    }
    stop();
}

void Worker::pushJob(const Job& job)
{
    std::lock_guard guard(impl->queue_mutex);
    impl->jobs.push(job);
}

void Worker::start()
{
    if(!impl->running)
    {
        impl->running = true;

        impl->worker_thread = std::thread(&Impl::worker_loop, impl.get());
    }
}

void Worker::startSync()
{
    start();
    waitDone();
}

bool Worker::isRunning() const
{
    return impl->running;
}

void Worker::stop()
{
    if(impl->running)
    {
        impl->running = false;
        if(impl->worker_thread.joinable()) impl->worker_thread.join();
    }
}

bool Worker::isDone() const
{
    std::lock_guard guard(impl->queue_mutex);
    return impl->jobs.empty() && !impl->job_running;
}

void Worker::waitDone() const
{
    while(!isDone())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void Worker::clearQueue()
{
    std::lock_guard guard(impl->queue_mutex);
    std::queue<Job> empty;
    impl->jobs.swap(empty);
}

uint32_t Worker::pendingJobs() const
{
    std::lock_guard guard(impl->queue_mutex);
    return impl->jobs.size();
}
}    // namespace atcg