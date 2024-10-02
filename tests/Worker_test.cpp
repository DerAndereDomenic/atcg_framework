#include <gtest/gtest.h>
#include <DataStructure/Worker.h>
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