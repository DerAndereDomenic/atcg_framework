#include <gtest/gtest.h>
#include <Math/Functions.h>
#include <DataStructure/BufferView.h>

TEST(BufferViewTest, SameType)
{
    int test[3] = {0, 1, 2};

    atcg::BufferView<int> view(test, sizeof(test));

    EXPECT_EQ(view.size(), sizeof(test));
    EXPECT_EQ(view.numel(), 3);
    EXPECT_EQ(view.stride(), sizeof(int));
    EXPECT_EQ(view[0], 0);
    EXPECT_EQ(view[1], 1);
    EXPECT_EQ(view[2], 2);
}

TEST(BufferViewTest, DifferentType)
{
    int test[6] = {0, 1, 2, 3, 4, 5};

    atcg::BufferView<glm::ivec3> view(test, sizeof(test));

    EXPECT_EQ(view.size(), sizeof(test));
    EXPECT_EQ(view.numel(), 2);
    EXPECT_EQ(view.stride(), sizeof(glm::ivec3));
    EXPECT_EQ(view[0].x, 0);
    EXPECT_EQ(view[0].y, 1);
    EXPECT_EQ(view[0].z, 2);
    EXPECT_EQ(view[1].x, 3);
    EXPECT_EQ(view[1].y, 4);
    EXPECT_EQ(view[1].z, 5);
}

TEST(BufferViewTest, DoubleStride)
{
    int test[4] = {0, 1, 2, 4};

    atcg::BufferView<int> view(test, sizeof(test), 2 * sizeof(int));

    EXPECT_EQ(view.size(), sizeof(test));
    EXPECT_EQ(view.numel(), 2);
    EXPECT_EQ(view.stride(), 2 * sizeof(int));
    EXPECT_EQ(view[0], 0);
    EXPECT_EQ(view[1], 2);
}

TEST(BufferViewTest, PreIncSameType)
{
    int test[3] = {0, 1, 2};

    atcg::BufferView<int> view(test, sizeof(test));

    EXPECT_EQ(*view, 0);
    ++view;
    EXPECT_EQ(*view, 1);
    ++view;
    EXPECT_EQ(*view, 2);

    EXPECT_EQ(view.size(), sizeof(test));
    EXPECT_EQ(view.numel(), 3);
    EXPECT_EQ(view.stride(), sizeof(int));
}

TEST(BufferViewTest, PostIncSameType)
{
    int test[3] = {0, 1, 2};

    atcg::BufferView<int> view(test, sizeof(test));

    EXPECT_EQ(*view, 0);
    view++;
    EXPECT_EQ(*view, 1);
    view++;
    EXPECT_EQ(*view, 2);

    EXPECT_EQ(view.size(), sizeof(test));
    EXPECT_EQ(view.numel(), 3);
    EXPECT_EQ(view.stride(), sizeof(int));
}

TEST(BufferViewTest, PreIncDiffType)
{
    int test[6] = {0, 1, 2, 3, 4, 5};

    atcg::BufferView<glm::ivec3> view(test, sizeof(test));
    EXPECT_EQ((*view).x, 0);
    EXPECT_EQ((*view).y, 1);
    EXPECT_EQ((*view).z, 2);
    ++view;
    EXPECT_EQ((*view).x, 3);
    EXPECT_EQ((*view).y, 4);
    EXPECT_EQ((*view).z, 5);

    EXPECT_EQ(view.size(), sizeof(test));
    EXPECT_EQ(view.numel(), 2);
    EXPECT_EQ(view.stride(), sizeof(glm::ivec3));
}

TEST(BufferViewTest, PostIncDiffType)
{
    int test[6] = {0, 1, 2, 3, 4, 5};

    atcg::BufferView<glm::ivec3> view(test, sizeof(test));
    EXPECT_EQ((*view).x, 0);
    EXPECT_EQ((*view).y, 1);
    EXPECT_EQ((*view).z, 2);
    view++;
    EXPECT_EQ((*view).x, 3);
    EXPECT_EQ((*view).y, 4);
    EXPECT_EQ((*view).z, 5);

    EXPECT_EQ(view.size(), sizeof(test));
    EXPECT_EQ(view.numel(), 2);
    EXPECT_EQ(view.stride(), sizeof(glm::ivec3));
}

TEST(BufferViewTest, ResetSameType)
{
    int test[6] = {0, 1, 2, 3, 4, 5};

    atcg::BufferView<int> view(test, sizeof(test));
    EXPECT_EQ(*view, 0);
    view++;
    view++;
    view++;
    EXPECT_EQ(*view, 3);
    view.reset();
    EXPECT_EQ(*view, 0);

    EXPECT_EQ(view.size(), sizeof(test));
    EXPECT_EQ(view.numel(), 6);
    EXPECT_EQ(view.stride(), sizeof(int));
}

TEST(BufferViewTest, DiffSameType)
{
    int test[6] = {0, 1, 2, 3, 4, 5};

    atcg::BufferView<glm::ivec3> view(test, sizeof(test));
    EXPECT_EQ((*view).x, 0);
    EXPECT_EQ((*view).y, 1);
    EXPECT_EQ((*view).z, 2);
    view++;
    EXPECT_EQ((*view).x, 3);
    EXPECT_EQ((*view).y, 4);
    EXPECT_EQ((*view).z, 5);
    view.reset();
    EXPECT_EQ((*view).x, 0);
    EXPECT_EQ((*view).y, 1);
    EXPECT_EQ((*view).z, 2);

    EXPECT_EQ(view.size(), sizeof(test));
    EXPECT_EQ(view.numel(), 2);
    EXPECT_EQ(view.stride(), sizeof(glm::ivec3));
}