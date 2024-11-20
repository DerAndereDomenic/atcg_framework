#include <gtest/gtest.h>
#include <DataStructure/Worker.h>
#include <DataStructure/WorkerPool.h>
#include <thread>
#include <chrono>

TEST(WorkerTest, start)
{
    atcg::Worker worker;

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    worker.start();
    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, startStop)
{
    atcg::Worker worker;

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    worker.start();
    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());

    worker.stop();
    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());
}

TEST(WorkerTest, simplePreTask)
{
    atcg::Worker worker;

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    worker.pushJob([]() { EXPECT_TRUE(true); });

    EXPECT_FALSE(worker.isDone());

    worker.start();

    worker.waitDone();
    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, simplePostTask)
{
    atcg::Worker worker;

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    worker.start();
    worker.pushJob([]() { EXPECT_TRUE(true); });

    worker.waitDone();
    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, multiJob)
{
    atcg::Worker worker;

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    int job_id = 0;
    worker.pushJob([&job_id]() { EXPECT_EQ(job_id++, 0); });
    worker.pushJob([&job_id]() { EXPECT_EQ(job_id++, 1); });
    worker.pushJob([&job_id]() { EXPECT_EQ(job_id++, 2); });
    worker.pushJob([&job_id]() { EXPECT_EQ(job_id++, 3); });
    worker.start();

    worker.waitDone();
    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, multiJobPost)
{
    atcg::Worker worker;

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    worker.start();
    int job_id = 0;
    worker.pushJob([&job_id]() { EXPECT_EQ(job_id++, 0); });
    worker.pushJob([&job_id]() { EXPECT_EQ(job_id++, 1); });
    worker.pushJob([&job_id]() { EXPECT_EQ(job_id++, 2); });
    worker.pushJob([&job_id]() { EXPECT_EQ(job_id++, 3); });

    worker.waitDone();
    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, longJob)
{
    atcg::Worker worker;

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());


    bool job_running = true;
    worker.start();
    worker.pushJob(
        [&job_running]()
        {
            while(job_running)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });

    EXPECT_FALSE(worker.isDone());
    job_running = false;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_TRUE(worker.isDone());

    worker.waitDone();
    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, longJobSync)
{
    atcg::Worker worker;

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.startSync();

    EXPECT_TRUE(worker.isDone());

    worker.waitDone();
    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, longMultiJobSync)
{
    atcg::Worker worker;

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.startSync();

    EXPECT_TRUE(worker.isDone());

    worker.waitDone();
    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, longMultiJob)
{
    atcg::Worker worker;

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.start();

    EXPECT_FALSE(worker.isDone());

    worker.waitDone();
    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, multiWorker)
{
    atcg::Worker workers[5];

    for(atcg::Worker& worker: workers)
    {
        EXPECT_TRUE(worker.isDone());
        EXPECT_FALSE(worker.isRunning());

        worker.start();
        worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    }

    for(atcg::Worker& worker: workers)
    {
        worker.waitDone();
        EXPECT_TRUE(worker.isDone());
        EXPECT_TRUE(worker.isRunning());
    }
}


TEST(WorkerTest, stopWhileRun)
{
    atcg::Worker worker;

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    worker.start();
    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });

    std::this_thread::sleep_for(std::chrono::milliseconds(500));    // Make sure job actually started
    EXPECT_FALSE(worker.isDone());
    worker.stop();
    EXPECT_FALSE(worker.isRunning());
    EXPECT_TRUE(worker.isDone());
}

TEST(WorkerTest, startPool)
{
    atcg::WorkerPool worker(4);

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    worker.start();
    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, startStopPool)
{
    atcg::WorkerPool worker(4);

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    worker.start();
    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());

    worker.stop();
    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());
}

TEST(WorkerTest, simplePreTaskPool)
{
    atcg::WorkerPool worker(4);

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    worker.pushJob([]() { EXPECT_TRUE(true); });

    EXPECT_FALSE(worker.isDone());

    worker.start();

    worker.waitDone();
    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, simplePostTaskPool)
{
    atcg::WorkerPool worker(4);

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    worker.start();
    worker.pushJob([]() { EXPECT_TRUE(true); });

    worker.waitDone();
    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, multiJobPrePool)
{
    atcg::WorkerPool worker(4);

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    worker.start();
    uint32_t job1, job2, job3, job4;
    worker.pushJob([&job1]() { job1 = 1; });
    worker.pushJob([&job2]() { job2 = 2; });
    worker.pushJob([&job3]() { job3 = 3; });
    worker.pushJob([&job4]() { job4 = 4; });

    worker.waitDone();

    EXPECT_EQ(job1, 1);
    EXPECT_EQ(job2, 2);
    EXPECT_EQ(job3, 3);
    EXPECT_EQ(job4, 4);

    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, multiJobPostPool)
{
    atcg::WorkerPool worker(4);

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    uint32_t job1, job2, job3, job4;
    worker.pushJob([&job1]() { job1 = 1; });
    worker.pushJob([&job2]() { job2 = 2; });
    worker.pushJob([&job3]() { job3 = 3; });
    worker.pushJob([&job4]() { job4 = 4; });
    worker.start();

    worker.waitDone();

    EXPECT_EQ(job1, 1);
    EXPECT_EQ(job2, 2);
    EXPECT_EQ(job3, 3);
    EXPECT_EQ(job4, 4);

    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, multiJobBetweenPool)
{
    atcg::WorkerPool worker(4);

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    uint32_t job1, job2, job3, job4;
    worker.pushJob([&job1]() { job1 = 1; });
    worker.pushJob([&job2]() { job2 = 2; });
    worker.start();
    worker.pushJob([&job3]() { job3 = 3; });
    worker.pushJob([&job4]() { job4 = 4; });

    worker.waitDone();

    EXPECT_EQ(job1, 1);
    EXPECT_EQ(job2, 2);
    EXPECT_EQ(job3, 3);
    EXPECT_EQ(job4, 4);

    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, multiJobPool)
{
    atcg::WorkerPool worker(12);

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    uint32_t results[1000];
    worker.start();

    for(int i = 0; i < 1000; ++i)
    {
        worker.pushJob(
            [&results, i]()
            {
                results[i] = i;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            });
    }
    EXPECT_FALSE(worker.isDone());

    worker.waitDone();

    for(int i = 0; i < 1000; ++i)
    {
        EXPECT_EQ(results[i], i);
    }

    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, longJobPool)
{
    atcg::WorkerPool worker(4);

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    bool job_running = true;
    worker.start();
    worker.pushJob(
        [&job_running]()
        {
            while(job_running)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });

    EXPECT_FALSE(worker.isDone());
    job_running = false;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_TRUE(worker.isDone());

    worker.waitDone();
    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, longJobSyncPool)
{
    atcg::WorkerPool worker(4);

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.startSync();

    EXPECT_TRUE(worker.isDone());

    worker.waitDone();
    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, longMultiJobSyncPool)
{
    atcg::WorkerPool worker(4);

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.startSync();

    EXPECT_TRUE(worker.isDone());

    worker.waitDone();
    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, longMultiJobPool)
{
    atcg::WorkerPool worker(4);

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });
    worker.start();

    EXPECT_FALSE(worker.isDone());

    worker.waitDone();
    EXPECT_TRUE(worker.isDone());
    EXPECT_TRUE(worker.isRunning());
}

TEST(WorkerTest, stopWhileRunPool)
{
    atcg::WorkerPool worker(4);

    EXPECT_TRUE(worker.isDone());
    EXPECT_FALSE(worker.isRunning());

    worker.start();
    worker.pushJob([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); });

    std::this_thread::sleep_for(std::chrono::milliseconds(500));    // Make sure job actually started
    EXPECT_FALSE(worker.isDone());
    worker.stop();
    EXPECT_FALSE(worker.isRunning());
    EXPECT_TRUE(worker.isDone());
}