#pragma once

namespace atcg
{

namespace Math
{

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T
map(const T& value, const T& old_left, const T& old_right, const T& new_left, const T& new_right)
{
    T m = (new_right - new_left) / (old_right - old_left);
    T b = new_left - m * old_left;

    return m * value + b;
}

namespace detail
{

template<typename glm::length_t N, typename T>
struct dispatch_map
{
    static ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec<N, T> apply(const glm::vec<N, T>& value,
                                                                   const glm::vec<N, T>& old_left,
                                                                   const glm::vec<N, T>& old_right,
                                                                   const glm::vec<N, T>& new_left,
                                                                   const glm::vec<N, T>& new_right)
    {
        return glm::vec<N, T>(0);
    }
};

template<typename T>
struct dispatch_map<1, T>
{
    static ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec<1, T> apply(const glm::vec<1, T>& value,
                                                                   const glm::vec<1, T>& old_left,
                                                                   const glm::vec<1, T>& old_right,
                                                                   const glm::vec<1, T>& new_left,
                                                                   const glm::vec<1, T>& new_right)
    {
        return glm::vec<1, T>(map(value, old_left, old_right, new_left, new_right));
    }
};

template<typename T>
struct dispatch_map<2, T>
{
    static ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec<2, T> apply(const glm::vec<2, T>& value,
                                                                   const glm::vec<2, T>& old_left,
                                                                   const glm::vec<2, T>& old_right,
                                                                   const glm::vec<2, T>& new_left,
                                                                   const glm::vec<2, T>& new_right)
    {
        return glm::vec<2, T>(map(value.x, old_left.x, old_right.x, new_left.x, new_right.x),
                              map(value.y, old_left.y, old_right.y, new_left.y, new_right.y));
    }
};

template<typename T>
struct dispatch_map<3, T>
{
    static ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec<3, T> apply(const glm::vec<3, T>& value,
                                                                   const glm::vec<3, T>& old_left,
                                                                   const glm::vec<3, T>& old_right,
                                                                   const glm::vec<3, T>& new_left,
                                                                   const glm::vec<3, T>& new_right)
    {
        return glm::vec<3, T>(map(value.x, old_left.x, old_right.x, new_left.x, new_right.x),
                              map(value.y, old_left.y, old_right.y, new_left.y, new_right.y),
                              map(value.z, old_left.z, old_right.z, new_left.z, new_right.z));
    }
};

template<typename T>
struct dispatch_map<4, T>
{
    static ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec<4, T> apply(const glm::vec<4, T>& value,
                                                                   const glm::vec<4, T>& old_left,
                                                                   const glm::vec<4, T>& old_right,
                                                                   const glm::vec<4, T>& new_left,
                                                                   const glm::vec<4, T>& new_right)
    {
        return glm::vec<4, T>(map(value.x, old_left.x, old_right.x, new_left.x, new_right.x),
                              map(value.y, old_left.y, old_right.y, new_left.y, new_right.y),
                              map(value.z, old_left.z, old_right.z, new_left.z, new_right.z),
                              map(value.w, old_left.w, old_right.w, new_left.w, new_right.w));
    }
};
}    // namespace detail

template<glm::length_t N, typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec<N, T> map(const glm::vec<N, T>& value,
                                                      const glm::vec<N, T>& old_left,
                                                      const glm::vec<N, T>& old_right,
                                                      const glm::vec<N, T>& new_left,
                                                      const glm::vec<N, T>& new_right)
{
    return detail::dispatch_map<N, T>::apply(value, old_left, old_right, new_left, new_right);
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T uv2ndc(const T& val)
{
    return T(2) * val - T(1);
}

template<glm::length_t N, typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec<N, T> uv2ndc(const glm::vec<N, T>& val)
{
    return glm::vec<N, T>(2) * val - glm::vec<N, T>(1);
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T ndc2uv(const T& val)
{
    return T(0.5) * val + T(0.5);
}

template<glm::length_t N, typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec<N, T> ndc2uv(const glm::vec<N, T>& val)
{
    return glm::vec<N, T>(0.5) * val + glm::vec<N, T>(0.5);
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T ndc2linearDepth(const T& ndc_depth, const T& n, const T& f)
{
    return (T(2) * n * f) / (f + n - ndc_depth * (f - n));
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T linearDepth2ndc(const T& linear_depth, const T& n, const T& f)
{
    return (f + n) / (f - n) - (T(2) * f * n / (f - n)) / linear_depth;
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::mat3 compute_local_frame(const glm::vec3& localZ)
{
    float x  = localZ.x;
    float y  = localZ.y;
    float z  = localZ.z;
    float sz = (z >= 0) ? 1 : -1;
    float a  = 1 / (sz + z);
    float ya = y * a;
    float b  = x * ya;
    float c  = x * sz;

    glm::vec3 localX = glm::vec3(c * x * a - 1, sz * b, c);
    glm::vec3 localY = glm::vec3(b, y * ya - sz, y);

    glm::mat3 frame;
    // Set columns of matrix
    frame[0] = localX;
    frame[1] = localY;
    frame[2] = localZ;
    return frame;
}

}    // namespace Math
}    // namespace atcg