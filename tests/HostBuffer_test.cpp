#include <gtest/gtest.h>
#include <Core/Memory.h>

// Demonstrate some basic assertions.
TEST(HostBufferTest, DefaultConstructor)
{
    atcg::DeviceBuffer<int> a;

    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(a.capacity(), 0);
    EXPECT_EQ(a.operator bool(), false);
}

TEST(HostBufferTest, AllocConstructor)
{
    atcg::DeviceBuffer<int> a(5);

    EXPECT_EQ(a.size(), 5);
    EXPECT_EQ(a.capacity(), 5);
    EXPECT_EQ(a.operator bool(), true);
}

TEST(HostBufferTest, Resize)
{
    atcg::DeviceBuffer<int> a(5);

    a.create(10);

    EXPECT_EQ(a.size(), 10);
    EXPECT_EQ(a.capacity(), 10);
    EXPECT_EQ(a.operator bool(), true);

    a.create(5);

    EXPECT_EQ(a.size(), 5);
    EXPECT_EQ(a.capacity(), 10);
    EXPECT_EQ(a.operator bool(), true);
}

TEST(HostBufferTest, NullptrConstructor)
{
    atcg::DeviceBuffer<int> a = nullptr;

    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(a.capacity(), 0);
    EXPECT_EQ(a.operator bool(), false);
}

TEST(HostBufferTest, PtrConstructor)
{
    int* i = new int;
    atcg::DeviceBuffer<int> a(i);

    EXPECT_EQ(a.size(), 1);
    EXPECT_EQ(a.capacity(), 1);
    EXPECT_EQ(a.operator bool(), true);
}

TEST(HostBufferTest, PtrConstructorBuffer)
{
    int* i = new int[4];
    atcg::DeviceBuffer<int> a(i, 4);

    EXPECT_EQ(a.size(), 4);
    EXPECT_EQ(a.capacity(), 4);
    EXPECT_EQ(a.operator bool(), true);
}

TEST(HostBufferTest, Destructor)
{
    atcg::host_allocator alloc;
    int alloc_before   = alloc.bytes_allocated;
    int dealloc_before = alloc.bytes_deallocated;
    {
        atcg::DeviceBuffer<int> a(5);
    }


    EXPECT_EQ(alloc.bytes_allocated - alloc_before, 5 * sizeof(int));
    EXPECT_EQ(alloc.bytes_deallocated - dealloc_before, 5 * sizeof(int));
}

TEST(HostBufferTest, DestructorCopy)
{
    atcg::host_allocator alloc;
    int alloc_before   = alloc.bytes_allocated;
    int dealloc_before = alloc.bytes_deallocated;

    atcg::DeviceBuffer<int> a(1);
    {
        atcg::DeviceBuffer<int> b = a;
    }


    EXPECT_EQ(alloc.bytes_allocated - alloc_before, sizeof(int));
    EXPECT_EQ(alloc.bytes_deallocated - dealloc_before, 0);
}

TEST(HostBufferTest, UseCount)
{
    atcg::DeviceBuffer<int> a(1);
    EXPECT_EQ(a.use_count(), 1);

    atcg::DeviceBuffer<int> b = a;
    EXPECT_EQ(a.use_count(), 2);

    {
        atcg::DeviceBuffer<int> c = a;
        EXPECT_EQ(a.use_count(), 3);
    }

    EXPECT_EQ(a.use_count(), 2);
}

TEST(HostBufferTest, UploadEqual)
{
    atcg::DeviceBuffer<int> a(5);

    std::vector<int> data(5);

    for(int i = 0; i < data.size(); ++i) { data[i] = i; }

    a.upload(data.data());

    int* a_ptr = a.get();
    for(int i = 0; i < data.size(); ++i) { EXPECT_EQ(a_ptr[i], i); }
}

TEST(HostBufferTest, UploadLess)
{
    atcg::DeviceBuffer<int> a(5);

    std::vector<int> data(3);

    for(int i = 0; i < data.size(); ++i) { data[i] = i; }

    a.upload(data.data());

    int* a_ptr = a.get();
    for(int i = 0; i < data.size(); ++i) { EXPECT_EQ(a_ptr[i], i); }

    EXPECT_EQ(a.capacity(), 5);
    EXPECT_EQ(a.size(), 5);
}

TEST(HostBufferTest, UploadMore)
{
    atcg::DeviceBuffer<int> a(5);

    std::vector<int> data(10);

    for(int i = 0; i < data.size(); ++i) { data[i] = i; }

    EXPECT_EQ(a.capacity(), 5);
    EXPECT_EQ(a.size(), 5);
    a.upload(data.data(), 10);

    int* a_ptr = a.get();
    for(int i = 0; i < data.size(); ++i) { EXPECT_EQ(a_ptr[i], i); }

    EXPECT_EQ(a.capacity(), 10);
    EXPECT_EQ(a.size(), 10);
}

TEST(HostBufferTest, DownloadEqual)
{
    int n = 5;
    atcg::DeviceBuffer<int> a(n);
    int* a_ptr = a.get();
    for(int i = 0; i < n; ++i) { a_ptr[i] = i; }

    std::vector<int> data(5);


    a.download(data.data());

    for(int i = 0; i < data.size(); ++i) { EXPECT_EQ(data[i], i); }
}

TEST(HostBufferTest, DownloadLess)
{
    int n = 5;
    atcg::DeviceBuffer<int> a(n);
    int* a_ptr = a.get();
    for(int i = 0; i < n; ++i) { a_ptr[i] = i; }

    std::vector<int> data(10);


    a.download(data.data());

    for(int i = 0; i < n; ++i) { EXPECT_EQ(data[i], i); }
}

// Demonstrate some basic assertions.
TEST(HostPointerTest, DefaultConstructor)
{
    atcg::ref_ptr<int> a;

    EXPECT_EQ(a.operator bool(), false);
    EXPECT_EQ(a.get(), nullptr);
}

TEST(HostPointerTest, NullptrConstructor)
{
    atcg::ref_ptr<int> a = nullptr;

    EXPECT_EQ(a.get(), nullptr);
    EXPECT_EQ(a.operator bool(), false);
}

TEST(HostPointerTest, PtrConstructor)
{
    int* i = new int;
    *i     = 5;
    atcg::ref_ptr<int> a(i);

    EXPECT_EQ(*a.get(), 5);
    EXPECT_EQ(a.operator bool(), true);
}

TEST(HostPointerTest, Destructor)
{
    atcg::host_allocator alloc;
    int alloc_before   = alloc.bytes_allocated;
    int dealloc_before = alloc.bytes_deallocated;
    {
        atcg::ref_ptr<int> a = atcg::make_ref<int>(5);
    }


    EXPECT_EQ(alloc.bytes_allocated - alloc_before, sizeof(int));
    EXPECT_EQ(alloc.bytes_deallocated - dealloc_before, sizeof(int));
}

TEST(HostPointerTest, DestructorCopy)
{
    atcg::host_allocator alloc;
    int alloc_before   = alloc.bytes_allocated;
    int dealloc_before = alloc.bytes_deallocated;

    atcg::ref_ptr<int> a = atcg::make_ref<int>(1);
    {
        atcg::ref_ptr<int> b = a;
    }


    EXPECT_EQ(alloc.bytes_allocated - alloc_before, sizeof(int));
    EXPECT_EQ(alloc.bytes_deallocated - dealloc_before, 0);
}

TEST(HostPointerTest, UseCount)
{
    atcg::ref_ptr<int> a;
    EXPECT_EQ(a.use_count(), 1);

    atcg::ref_ptr<int> b = a;
    EXPECT_EQ(a.use_count(), 2);

    {
        atcg::ref_ptr<int> c = a;
        EXPECT_EQ(a.use_count(), 3);
    }

    EXPECT_EQ(a.use_count(), 2);
}

TEST(HostPointerTest, UploadEqual)
{
    atcg::ref_ptr<int> a;

    int data = 6;

    a.upload(&data);

    int* a_ptr = a.get();
    EXPECT_EQ(*a_ptr, 6);
}

TEST(HostPointerTest, DownloadEqual)
{
    atcg::ref_ptr<int> a = atcg::make_ref<int>(5);
    int* a_ptr           = a.get();

    int data;
    a.download(&data);

    EXPECT_EQ(data, 5);
}

TEST(HostPointerTest, make_shared)
{
    atcg::ref_ptr<int> a = atcg::make_ref<int>(5);

    EXPECT_EQ(*a.get(), 5);
    EXPECT_EQ(a.operator bool(), true);
}


TEST(HostPointerTest, Cast)
{
    class A
    {
    public:
        A() = default;

        int x;
    };

    class B : public A
    {
    public:
        B() = default;

        int y;
    };

    atcg::host_allocator alloc;
    int alloc_before   = alloc.bytes_allocated;
    int dealloc_before = alloc.bytes_deallocated;

    {
        atcg::ref_ptr<A> a = atcg::make_ref<B>();
    }

    EXPECT_EQ(alloc.bytes_allocated - alloc_before, sizeof(B));
    EXPECT_EQ(alloc.bytes_deallocated - dealloc_before, sizeof(B));
}