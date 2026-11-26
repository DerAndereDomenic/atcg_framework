#include "Bindings.h"

#include <sstream>
#include <type_traits>

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
namespace
{
/**
 * @brief Define the members that all glm vector types share: buffer protocol, numpy conversion, indexing, iteration,
 * representation and arithmetic operators.
 *
 * @tparam V The glm vector type
 * @tparam T The scalar type
 * @tparam N The number of components
 * @param cls The python class
 * @param name The python name of the type
 */
template<typename V, typename T, int N>
void defineVector(py::class_<V>& cls, const char* name)
{
    cls.def_buffer(
           [](V& v) -> py::buffer_info
           {
               return py::buffer_info(glm::value_ptr(v),
                                      (py::ssize_t)sizeof(T),
                                      py::format_descriptor<T>::format(),
                                      1,
                                      {(py::ssize_t)N},
                                      {(py::ssize_t)sizeof(T)});
           })
        .def(py::init(
                 [name](py::array_t<T, py::array::c_style | py::array::forcecast> b)
                 {
                     V v(T(0));
                     if(b.size() != N)
                     {
                         ATCG_ERROR("{0}: Expected an array with {1} elements but got {2}", name, N, b.size());
                         return v;
                     }

                     const T* data = b.data();
                     for(int i = 0; i < N; ++i)
                     {
                         v[i] = data[i];
                     }

                     return v;
                 }),
             "array"_a)
        .def("numpy",
             [](const V& v) -> py::array_t<T>
             { return py::array_t<T>({(py::ssize_t)N}, {(py::ssize_t)sizeof(T)}, glm::value_ptr(v)); })
        .def("__len__", [](const V& v) { return N; })
        .def(
            "__getitem__",
            [name](const V& v, int i) -> T
            {
                if(i < 0) i += N;
                if(i < 0 || i >= N)
                {
                    ATCG_ERROR("{0}: Index {1} out of range [0, {2})", name, i, N);
                    return T(0);
                }
                return v[i];
            },
            "index"_a)
        .def(
            "__setitem__",
            [name](V& v, int i, T value)
            {
                if(i < 0) i += N;
                if(i < 0 || i >= N)
                {
                    ATCG_ERROR("{0}: Index {1} out of range [0, {2})", name, i, N);
                    return;
                }
                v[i] = value;
            },
            "index"_a,
            "value"_a)
        .def(
            "__iter__",
            [](const V& v) { return py::make_iterator(glm::value_ptr(v), glm::value_ptr(v) + N); },
            py::keep_alive<0, 1>())
        .def("__repr__",
             [name](const V& v)
             {
                 std::ostringstream ss;
                 ss << name << "(";
                 for(int i = 0; i < N; ++i)
                 {
                     if(i > 0) ss << ", ";
                     ss << v[i];
                 }
                 ss << ")";
                 return ss.str();
             })
        .def(
            "__eq__",
            [](const V& a, const V& b) { return a == b; },
            py::is_operator())
        .def(
            "__ne__",
            [](const V& a, const V& b) { return a != b; },
            py::is_operator())
        .def(
            "__add__",
            [](const V& a, const V& b) { return V(a + b); },
            py::is_operator())
        .def(
            "__add__",
            [](const V& a, T s) { return V(a + s); },
            py::is_operator())
        .def(
            "__radd__",
            [](const V& a, T s) { return V(s + a); },
            py::is_operator())
        .def(
            "__sub__",
            [](const V& a, const V& b) { return V(a - b); },
            py::is_operator())
        .def(
            "__sub__",
            [](const V& a, T s) { return V(a - s); },
            py::is_operator())
        .def(
            "__rsub__",
            [](const V& a, T s) { return V(s - a); },
            py::is_operator())
        .def(
            "__mul__",
            [](const V& a, const V& b) { return V(a * b); },
            py::is_operator())
        .def(
            "__mul__",
            [](const V& a, T s) { return V(a * s); },
            py::is_operator())
        .def(
            "__rmul__",
            [](const V& a, T s) { return V(s * a); },
            py::is_operator());

    if constexpr(std::is_floating_point_v<T>)
    {
        cls.def(
               "__truediv__",
               [](const V& a, const V& b) { return V(a / b); },
               py::is_operator())
            .def(
                "__truediv__",
                [](const V& a, T s) { return V(a / s); },
                py::is_operator())
            .def(
                "__rtruediv__",
                [](const V& a, T s) { return V(s / a); },
                py::is_operator());
    }
    else
    {
        // Integer division follows the C++ semantics (truncation towards zero)
        cls.def(
               "__floordiv__",
               [name](const V& a, const V& b)
               {
                   for(int i = 0; i < N; ++i)
                   {
                       if(b[i] == T(0))
                       {
                           ATCG_ERROR("{0}: Division by zero", name);
                           return V(T(0));
                       }
                   }
                   return V(a / b);
               },
               py::is_operator())
            .def(
                "__floordiv__",
                [name](const V& a, T s)
                {
                    if(s == T(0))
                    {
                        ATCG_ERROR("{0}: Division by zero", name);
                        return V(T(0));
                    }
                    return V(a / s);
                },
                py::is_operator())
            .def(
                "__rfloordiv__",
                [name](const V& a, T s)
                {
                    for(int i = 0; i < N; ++i)
                    {
                        if(a[i] == T(0))
                        {
                            ATCG_ERROR("{0}: Division by zero", name);
                            return V(T(0));
                        }
                    }
                    return V(s / a);
                },
                py::is_operator());
    }

    if constexpr(std::is_signed_v<T>)
    {
        cls.def(
            "__neg__",
            [](const V& a) { return V(-a); },
            py::is_operator());
    }
}

/**
 * @brief Define the members that all square glm matrix types share. The python side uses a row-major view of the
 * matrix, i.e. M[row, col]. glm stores the matrix in column-major order.
 *
 * @tparam M The glm matrix type
 * @tparam V The glm column vector type
 * @tparam N The number of rows and columns
 * @param cls The python class
 * @param name The python name of the type
 */
template<typename M, typename V, int N>
void defineMatrix(py::class_<M>& cls, const char* name)
{
    cls.def_buffer(
           [](M& mat) -> py::buffer_info
           {
               return py::buffer_info(glm::value_ptr(mat),
                                      (py::ssize_t)sizeof(float),
                                      py::format_descriptor<float>::format(),
                                      2,
                                      {(py::ssize_t)N, (py::ssize_t)N},
                                      {(py::ssize_t)sizeof(float), (py::ssize_t)(sizeof(float) * N)});
           })
        .def(py::init([]() { return M(1.0f); }))
        .def(py::init([](float value) { return M(value); }), "value"_a)
        .def(py::init(
                 [name](py::array_t<float, py::array::c_style | py::array::forcecast> b)
                 {
                     M mat(1.0f);
                     if(b.ndim() != 2 || b.shape(0) != N || b.shape(1) != N)
                     {
                         ATCG_ERROR("{0}: Expected an array of shape ({1}, {1})", name, N);
                         return mat;
                     }

                     const float* data = b.data();
                     for(int i = 0; i < N; ++i)
                     {
                         for(int j = 0; j < N; ++j)
                         {
                             mat[i][j] = data[b.index_at(j, i)];
                         }
                     }

                     return mat;
                 }),
             "array"_a)
        .def("numpy",
             [](const M& mat) -> py::array_t<float>
             {
                 float data[N][N];

                 for(int i = 0; i < N; ++i)
                 {
                     for(int j = 0; j < N; ++j)
                     {
                         data[i][j] = mat[j][i];
                     }
                 }

                 return py::array_t<float>({(py::ssize_t)N, (py::ssize_t)N},
                                           {(py::ssize_t)(sizeof(float) * N), (py::ssize_t)sizeof(float)},
                                           (const float*)data);
             })
        .def("__len__", [](const M& mat) { return N; })
        .def(
            "__getitem__",
            [name](const M& mat, const std::pair<int, int>& index) -> float
            {
                int row = index.first < 0 ? index.first + N : index.first;
                int col = index.second < 0 ? index.second + N : index.second;
                if(row < 0 || row >= N || col < 0 || col >= N)
                {
                    ATCG_ERROR("{0}: Index ({1}, {2}) out of range", name, index.first, index.second);
                    return 0.0f;
                }
                return mat[col][row];
            },
            "index"_a)
        .def(
            "__setitem__",
            [name](M& mat, const std::pair<int, int>& index, float value)
            {
                int row = index.first < 0 ? index.first + N : index.first;
                int col = index.second < 0 ? index.second + N : index.second;
                if(row < 0 || row >= N || col < 0 || col >= N)
                {
                    ATCG_ERROR("{0}: Index ({1}, {2}) out of range", name, index.first, index.second);
                    return;
                }
                mat[col][row] = value;
            },
            "index"_a,
            "value"_a)
        .def("__repr__",
             [name](const M& mat)
             {
                 std::ostringstream ss;
                 ss << name << "(";
                 for(int i = 0; i < N; ++i)
                 {
                     if(i > 0) ss << ", ";
                     ss << "(";
                     for(int j = 0; j < N; ++j)
                     {
                         if(j > 0) ss << ", ";
                         ss << mat[j][i];
                     }
                     ss << ")";
                 }
                 ss << ")";
                 return ss.str();
             })
        .def(
            "__eq__",
            [](const M& a, const M& b) { return a == b; },
            py::is_operator())
        .def(
            "__ne__",
            [](const M& a, const M& b) { return a != b; },
            py::is_operator())
        .def(
            "__add__",
            [](const M& a, const M& b) { return M(a + b); },
            py::is_operator())
        .def(
            "__sub__",
            [](const M& a, const M& b) { return M(a - b); },
            py::is_operator())
        .def(
            "__mul__",
            [](const M& a, const M& b) { return M(a * b); },
            py::is_operator())
        .def(
            "__mul__",
            [](const M& a, const V& v) { return V(a * v); },
            py::is_operator())
        .def(
            "__mul__",
            [](const M& a, float s) { return M(a * s); },
            py::is_operator())
        .def(
            "__rmul__",
            [](const M& a, float s) { return M(s * a); },
            py::is_operator())
        .def(
            "__truediv__",
            [](const M& a, float s) { return M(a / s); },
            py::is_operator())
        .def(
            "__neg__",
            [](const M& a) { return M(-a); },
            py::is_operator());
}
}    // namespace

// glm vector and matrix types
void bindMath(BindingContext& ctx)
{
    py::module_ m = ctx.module();

    auto m_vec2    = py::class_<glm::vec2>(m, "vec2", py::buffer_protocol());
    auto m_ivec2   = py::class_<glm::ivec2>(m, "ivec2", py::buffer_protocol());
    auto m_vec3    = py::class_<glm::vec3>(m, "vec3", py::buffer_protocol());
    auto m_ivec3   = py::class_<glm::ivec3>(m, "ivec3", py::buffer_protocol());
    auto m_u32vec3 = py::class_<glm::u32vec3>(m, "u32vec3", py::buffer_protocol());
    auto m_vec4    = py::class_<glm::vec4>(m, "vec4", py::buffer_protocol());
    auto m_ivec4   = py::class_<glm::ivec4>(m, "ivec4", py::buffer_protocol());
    auto m_mat3    = py::class_<glm::mat3>(m, "mat3", py::buffer_protocol());
    auto m_mat4    = py::class_<glm::mat4>(m, "mat4", py::buffer_protocol());

    ctx.define(
        [=]() mutable
        {
            // The scalar constructors have to be defined before the array constructors. Otherwise, a python scalar
            // would be converted into an array.
            m_vec2.def(py::init<float, float>(), "x"_a, "y"_a)
                .def(py::init<float>(), "value"_a)
                .def_readwrite("x", &glm::vec2::x)
                .def_readwrite("y", &glm::vec2::y);
            defineVector<glm::vec2, float, 2>(m_vec2, "vec2");

            m_ivec2.def(py::init<int, int>(), "x"_a, "y"_a)
                .def(py::init<int>(), "value"_a)
                .def_readwrite("x", &glm::ivec2::x)
                .def_readwrite("y", &glm::ivec2::y);
            defineVector<glm::ivec2, int, 2>(m_ivec2, "ivec2");

            m_vec3.def(py::init<float, float, float>(), "x"_a, "y"_a, "z"_a)
                .def(py::init<float>(), "value"_a)
                .def_readwrite("x", &glm::vec3::x)
                .def_readwrite("y", &glm::vec3::y)
                .def_readwrite("z", &glm::vec3::z);
            defineVector<glm::vec3, float, 3>(m_vec3, "vec3");

            m_ivec3.def(py::init<int, int, int>(), "x"_a, "y"_a, "z"_a)
                .def(py::init<int>(), "value"_a)
                .def_readwrite("x", &glm::ivec3::x)
                .def_readwrite("y", &glm::ivec3::y)
                .def_readwrite("z", &glm::ivec3::z);
            defineVector<glm::ivec3, int, 3>(m_ivec3, "ivec3");

            m_u32vec3.def(py::init<uint32_t, uint32_t, uint32_t>(), "x"_a, "y"_a, "z"_a)
                .def(py::init<uint32_t>(), "value"_a)
                .def_readwrite("x", &glm::u32vec3::x)
                .def_readwrite("y", &glm::u32vec3::y)
                .def_readwrite("z", &glm::u32vec3::z);
            defineVector<glm::u32vec3, uint32_t, 3>(m_u32vec3, "u32vec3");

            m_vec4.def(py::init<float, float, float, float>(), "x"_a, "y"_a, "z"_a, "w"_a)
                .def(py::init<float>(), "value"_a)
                .def_readwrite("x", &glm::vec4::x)
                .def_readwrite("y", &glm::vec4::y)
                .def_readwrite("z", &glm::vec4::z)
                .def_readwrite("w", &glm::vec4::w);
            defineVector<glm::vec4, float, 4>(m_vec4, "vec4");

            m_ivec4.def(py::init<int, int, int, int>(), "x"_a, "y"_a, "z"_a, "w"_a)
                .def(py::init<int>(), "value"_a)
                .def_readwrite("x", &glm::ivec4::x)
                .def_readwrite("y", &glm::ivec4::y)
                .def_readwrite("z", &glm::ivec4::z)
                .def_readwrite("w", &glm::ivec4::w);
            defineVector<glm::ivec4, int, 4>(m_ivec4, "ivec4");

            defineMatrix<glm::mat3, glm::vec3, 3>(m_mat3, "mat3");
            defineMatrix<glm::mat4, glm::vec4, 4>(m_mat4, "mat4");
        });
}
}    // namespace python
}    // namespace atcg
