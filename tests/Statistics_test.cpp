#include <gtest/gtest.h>
#include <DataStructure/Statistics.h>
#include <random>

std::vector<float> generateSamples(int n)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);    // distribution in range [1, 6]

    std::vector<float> samples(n);

    for(int i = 0; i < n; ++i)
    {
        samples[i] = uniform(rng);
    }

    return samples;
}

float mean(const std::vector<float>& samples)
{
    float m = 0.0f;
    for(float sample: samples)
    {
        m += sample;
    }

    return m / ((float)samples.size());
}

float var(const std::vector<float>& samples)
{
    float m = mean(samples);
    float v = 0.0f;
    for(float sample: samples)
    {
        v += (sample - m) * (sample - m);
    }

    return v / ((float)samples.size());
}

TEST(StatisticsTest, StatisticBase)
{
    auto samples = generateSamples(1000);

    atcg::Statistic<float> statistic;

    for(float sample: samples)
    {
        statistic.addSample(sample);
    }

    EXPECT_NEAR(statistic.mean(), mean(samples), 1e-5f);
    EXPECT_NEAR(statistic.var(), var(samples), 1e-5f);
    EXPECT_NEAR(statistic.count(), 1000, 1e-5f);
}

TEST(StatisticsTest, StatisticRemove)
{
    auto samples        = generateSamples(1000);
    auto samples_remove = generateSamples(1000);

    atcg::Statistic<float> statistic;

    for(float sample: samples)
    {
        statistic.addSample(sample);
    }

    for(float sample: samples_remove)
    {
        statistic.addSample(sample);
    }

    for(float sample: samples_remove)
    {
        statistic.removeSample(sample);
    }

    EXPECT_NEAR(statistic.mean(), mean(samples), 1e-5f);
    EXPECT_NEAR(statistic.var(), var(samples), 1e-5f);
}

TEST(StatisticsTest, StatisticReset)
{
    auto samples        = generateSamples(1000);
    auto samples_remove = generateSamples(1000);

    atcg::Statistic<float> statistic;

    for(float sample: samples)
    {
        statistic.addSample(sample);
    }

    EXPECT_NEAR(statistic.mean(), mean(samples), 1e-5f);
    EXPECT_NEAR(statistic.var(), var(samples), 1e-5f);
    statistic.reset();

    for(float sample: samples_remove)
    {
        statistic.addSample(sample);
    }


    EXPECT_NEAR(statistic.mean(), mean(samples_remove), 1e-5f);
    EXPECT_NEAR(statistic.var(), var(samples_remove), 1e-5f);
}

TEST(StatisticsTest, CollectionBase)
{
    auto samples = generateSamples(1000);

    atcg::Collection<float> collection("Collection", 1000);

    for(float sample: samples)
    {
        collection.addSample(sample);
    }

    EXPECT_NEAR(collection.mean(), mean(samples), 1e-5f);
    EXPECT_NEAR(collection.var(), var(samples), 1e-5f);
    EXPECT_NEAR(collection.count(), 1000, 1e-5f);

    for(int i = 0; i < 1000; ++i)
    {
        EXPECT_EQ((*collection)[i], samples[i]);
    }
}

TEST(StatisticsTest, CollectionReset)
{
    auto samples        = generateSamples(1000);
    auto samples_remove = generateSamples(1000);

    atcg::Collection<float> collection("Collection", 1000);

    for(float sample: samples)
    {
        collection.addSample(sample);
    }

    EXPECT_NEAR(collection.mean(), mean(samples), 1e-5f);
    EXPECT_NEAR(collection.var(), var(samples), 1e-5f);

    for(int i = 0; i < 1000; ++i)
    {
        EXPECT_EQ((*collection)[i], samples[i]);
    }

    collection.resetStatistics();

    for(float sample: samples_remove)
    {
        collection.addSample(sample);
    }


    EXPECT_NEAR(collection.mean(), mean(samples_remove), 1e-5f);
    EXPECT_NEAR(collection.var(), var(samples_remove), 1e-5f);

    for(int i = 0; i < 1000; ++i)
    {
        EXPECT_EQ((*collection)[i], samples_remove[i]);
    }
}

TEST(StatisticsTest, CollectionBaseNotFull)
{
    auto samples = generateSamples(1000);

    atcg::Collection<float> collection("Collection", 2000);

    for(float sample: samples)
    {
        collection.addSample(sample);
    }

    EXPECT_NEAR(collection.mean(), mean(samples), 1e-5f);
    EXPECT_NEAR(collection.var(), var(samples), 1e-5f);
    EXPECT_NEAR(collection.count(), 1000, 1e-5f);

    for(int i = 0; i < 1000; ++i)
    {
        EXPECT_EQ((*collection)[i], samples[i]);
    }
}

TEST(StatisticsTest, CollectionResetNotFull)
{
    auto samples        = generateSamples(1000);
    auto samples_remove = generateSamples(1000);

    atcg::Collection<float> collection("Collection", 2000);

    for(float sample: samples)
    {
        collection.addSample(sample);
    }

    EXPECT_NEAR(collection.mean(), mean(samples), 1e-5f);
    EXPECT_NEAR(collection.var(), var(samples), 1e-5f);

    for(int i = 0; i < 1000; ++i)
    {
        EXPECT_EQ((*collection)[i], samples[i]);
    }

    collection.resetStatistics();

    for(float sample: samples_remove)
    {
        collection.addSample(sample);
    }


    EXPECT_NEAR(collection.mean(), mean(samples_remove), 1e-5f);
    EXPECT_NEAR(collection.var(), var(samples_remove), 1e-5f);

    for(int i = 0; i < 1000; ++i)
    {
        EXPECT_EQ((*collection)[i], samples_remove[i]);
    }
}

TEST(StatisticsTest, CyclicCollectionBase)
{
    auto samples = generateSamples(1000);

    atcg::CyclicCollection<float> collection("Collection", 1000);

    for(float sample: samples)
    {
        collection.addSample(sample);
    }

    EXPECT_NEAR(collection.mean(), mean(samples), 1e-5f);
    EXPECT_NEAR(collection.var(), var(samples), 1e-5f);
    EXPECT_NEAR(collection.count(), 1000, 1e-5f);

    for(int i = 0; i < 1000; ++i)
    {
        EXPECT_EQ((*collection)[i], samples[i]);
    }
}

TEST(StatisticsTest, CyclicCollectionBaseNotFull)
{
    auto samples = generateSamples(1000);

    atcg::CyclicCollection<float> collection("Collection", 2000);

    for(float sample: samples)
    {
        collection.addSample(sample);
    }

    EXPECT_NEAR(collection.mean(), mean(samples), 1e-5f);
    EXPECT_NEAR(collection.var(), var(samples), 1e-5f);
    EXPECT_NEAR(collection.count(), 1000, 1e-5f);

    for(int i = 0; i < 1000; ++i)
    {
        EXPECT_EQ((*collection)[i], samples[i]);
    }
}

TEST(StatisticsTest, CyclicCollectionBaseOverFull)
{
    auto samples = generateSamples(500);

    atcg::CyclicCollection<float> collection("Collection", 500);

    for(float sample: samples)
    {
        collection.addSample(sample);
    }

    samples = generateSamples(500);
    for(float sample: samples)
    {
        collection.addSample(sample);
    }

    EXPECT_NEAR(collection.mean(), mean(samples), 1e-5f);
    EXPECT_NEAR(collection.var(), var(samples), 1e-5f);
    EXPECT_NEAR(collection.count(), 500, 1e-5f);

    for(int i = 0; i < 500; ++i)
    {
        EXPECT_EQ((*collection)[i], samples[i]);
    }
}

TEST(StatisticsTest, CyclicCollectionResetNotFull)
{
    auto samples        = generateSamples(1000);
    auto samples_remove = generateSamples(1000);

    atcg::CyclicCollection<float> collection("Collection", 2000);

    for(float sample: samples)
    {
        collection.addSample(sample);
    }

    EXPECT_NEAR(collection.mean(), mean(samples), 1e-5f);
    EXPECT_NEAR(collection.var(), var(samples), 1e-5f);

    for(int i = 0; i < 1000; ++i)
    {
        EXPECT_EQ((*collection)[i], samples[i]);
    }

    collection.resetStatistics();

    for(float sample: samples_remove)
    {
        collection.addSample(sample);
    }


    EXPECT_NEAR(collection.mean(), mean(samples_remove), 1e-5f);
    EXPECT_NEAR(collection.var(), var(samples_remove), 1e-5f);

    for(int i = 0; i < 1000; ++i)
    {
        EXPECT_EQ((*collection)[i], samples_remove[i]);
    }
}

TEST(StatisticsTest, CyclicCollectionResetOverFull)
{
    auto samples        = generateSamples(500);
    auto samples_remove = generateSamples(500);

    atcg::CyclicCollection<float> collection("Collection", 500);

    for(float sample: samples)
    {
        collection.addSample(sample);
    }

    for(float sample: samples_remove)
    {
        collection.addSample(sample);
    }

    EXPECT_NEAR(collection.mean(), mean(samples_remove), 1e-5f);
    EXPECT_NEAR(collection.var(), var(samples_remove), 1e-5f);

    for(int i = 0; i < 500; ++i)
    {
        EXPECT_EQ((*collection)[i], samples_remove[i]);
    }

    collection.resetStatistics();

    for(float sample: samples)
    {
        collection.addSample(sample);
    }


    EXPECT_NEAR(collection.mean(), mean(samples), 1e-5f);
    EXPECT_NEAR(collection.var(), var(samples), 1e-5f);

    for(int i = 0; i < 500; ++i)
    {
        EXPECT_EQ((*collection)[i], samples[i]);
    }
}