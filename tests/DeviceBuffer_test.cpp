#include <gtest/gtest.h>
#include <Core/Memory.h>

// Demonstrate some basic assertions.
TEST(DeviceBufferTest, DefaultConstructor)
{
    atcg::dref_ptr<int> a;

    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(a.capacity(), 0);
    EXPECT_EQ(a.operator bool(), false);
}

TEST(DeviceBufferTest, AllocConstructor)
{
    atcg::dref_ptr<int> a(5);

    EXPECT_EQ(a.size(), 5);
    EXPECT_EQ(a.capacity(), 5);
    EXPECT_EQ(a.operator bool(), true);
}

TEST(DeviceBufferTest, Resize)
{
    atcg::dref_ptr<int> a(5);

    a.create(10);

    EXPECT_EQ(a.size(), 10);
    EXPECT_EQ(a.capacity(), 10);
    EXPECT_EQ(a.operator bool(), true);

    a.create(5);

    EXPECT_EQ(a.size(), 5);
    EXPECT_EQ(a.capacity(), 10);
    EXPECT_EQ(a.operator bool(), true);
}

TEST(DeviceBufferTest, NullptrConstructor)
{
    atcg::dref_ptr<int> a = nullptr;

    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(a.capacity(), 0);
    EXPECT_EQ(a.operator bool(), false);
}

TEST(DeviceBufferTest, PtrConstructor)
{
    int* i;

#ifdef ATCG_CUDA_BACKEND
    cudaMalloc((void**)&i, sizeof(int) * 1);
#else
    i = new int;
#endif

    atcg::dref_ptr<int> a(i);

    EXPECT_EQ(a.size(), 1);
    EXPECT_EQ(a.capacity(), 1);
    EXPECT_EQ(a.operator bool(), true);
}

TEST(DeviceBufferTest, PtrConstructorBuffer)
{
    int* i;

#ifdef ATCG_CUDA_BACKEND
    cudaMalloc((void**)&i, sizeof(int) * 4);
#else
    i = new int;
#endif

    atcg::dref_ptr<int> a(i, 4);

    EXPECT_EQ(a.size(), 4);
    EXPECT_EQ(a.capacity(), 4);
    EXPECT_EQ(a.operator bool(), true);
}

TEST(DeviceBufferTest, Destructor)
{
    atcg::device_allocator alloc;
    int alloc_before   = alloc.bytes_allocated;
    int dealloc_before = alloc.bytes_deallocated;
    {
        atcg::dref_ptr<int> a(1);
    }


    EXPECT_EQ(alloc.bytes_allocated - alloc_before, sizeof(int));
    EXPECT_EQ(alloc.bytes_deallocated - dealloc_before, sizeof(int));
}

TEST(DeviceBufferTest, DestructorCopy)
{
    atcg::device_allocator alloc;
    int alloc_before   = alloc.bytes_allocated;
    int dealloc_before = alloc.bytes_deallocated;

    atcg::dref_ptr<int> a(1);
    {
        atcg::dref_ptr<int> b = a;
    }


    EXPECT_EQ(alloc.bytes_allocated - alloc_before, sizeof(int));
    EXPECT_EQ(alloc.bytes_deallocated - dealloc_before, 0);
}

TEST(DeviceBufferTest, UseCount)
{
    atcg::dref_ptr<int> a(1);
    EXPECT_EQ(a.use_count(), 1);

    atcg::dref_ptr<int> b = a;
    EXPECT_EQ(a.use_count(), 2);

    {
        atcg::dref_ptr<int> c = a;
        EXPECT_EQ(a.use_count(), 3);
    }

    EXPECT_EQ(a.use_count(), 2);
}

TEST(DeviceBufferTest, UploadEqual)
{
    atcg::dref_ptr<int> a(5);

    std::vector<int> data(5);

    for(int i = 0; i < data.size(); ++i) { data[i] = i; }

    a.upload(data.data());

    std::vector<int> downloaded_data(5);
    a.download(downloaded_data.data());
    for(int i = 0; i < downloaded_data.size(); ++i) { EXPECT_EQ(downloaded_data[i], i); }
}

TEST(DeviceBufferTest, UploadLess)
{
    atcg::dref_ptr<int> a(5);

    std::vector<int> data(3);

    for(int i = 0; i < data.size(); ++i) { data[i] = i; }

    a.upload(data.data());

    std::vector<int> downloaded_data(3);
    a.download(downloaded_data.data());
    for(int i = 0; i < downloaded_data.size(); ++i) { EXPECT_EQ(downloaded_data[i], i); }
}

TEST(DeviceBufferTest, UploadMore)
{
    atcg::dref_ptr<int> a(5);

    std::vector<int> data(10);

    for(int i = 0; i < data.size(); ++i) { data[i] = i; }

    EXPECT_EQ(a.capacity(), 5);
    EXPECT_EQ(a.size(), 5);
    a.upload(data.data(), 10);

    std::vector<int> downloaded_data(10);
    a.download(downloaded_data.data());
    for(int i = 0; i < downloaded_data.size(); ++i) { EXPECT_EQ(downloaded_data[i], i); }

    EXPECT_EQ(a.capacity(), 10);
    EXPECT_EQ(a.size(), 10);
}