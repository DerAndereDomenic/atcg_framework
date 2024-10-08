#include <DataStructure/WorkerPool.h>

namespace atcg
{

class WorkerPool::Impl
{
public:
    Impl(uint32_t num_workers);
    ~Impl();

    std::vector<Worker> worker_pool;
    bool running = false;
};

WorkerPool::Impl::Impl(uint32_t num_workers)
{
    worker_pool = std::vector<Worker>(num_workers);
}

WorkerPool::Impl::~Impl() {}

WorkerPool::WorkerPool(uint32_t num_workers)
{
    impl = std::make_unique<Impl>(num_workers);
}

WorkerPool::~WorkerPool() {}

void WorkerPool::pushJob(const Worker::Job& job)
{
    Worker* best_worker = nullptr;
    for(auto& worker: impl->worker_pool)
    {
        if(!best_worker || best_worker->pendingJobs() > worker.pendingJobs())
        {
            best_worker = &worker;
        }
    }

    best_worker->pushJob(job);
}

void WorkerPool::start()
{
    if(!impl->running)
    {
        for(auto& worker: impl->worker_pool)
        {
            worker.start();
        }
        impl->running = true;
    }
}

void WorkerPool::startSync()
{
    start();
    waitDone();
}

bool WorkerPool::isRunning() const
{
    return impl->running;
}

void WorkerPool::stop()
{
    if(impl->running)
    {
        for(auto& worker: impl->worker_pool)
        {
            worker.stop();
        }
        impl->running = false;
    }
}

bool WorkerPool::isDone() const
{
    bool done = true;
    for(auto& worker: impl->worker_pool)
    {
        done = done && worker.isDone();
    }
    return done;
    return false;
}

void WorkerPool::waitDone() const
{
    while(!isDone())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
}    // namespace atcg