#include "Bindings.h"

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
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
            m_vec2.def(py::init<float, float>(), "x"_a, "y"_a)
                .def(py::init<float>(), "value"_a)
                .def(py::init(
                         [](py::array_t<float> b)
                         {
                             py::buffer_info info = b.request();

                             // Copy for now, is there a better method?
                             glm::vec2 v = glm::make_vec2(static_cast<float*>(info.ptr));

                             return v;
                         }),
                     "array"_a)
                .def("numpy",
                     [](glm::vec2& v) -> py::array_t<float>
                     { return py::array_t<float>({2}, {sizeof(float)}, glm::value_ptr(v)); })
                .def_readwrite("x", &glm::vec2::x)
                .def_readwrite("y", &glm::vec2::y);

            m_ivec2.def(py::init<int, int>(), "x"_a, "y"_a)
                .def(py::init<int>(), "value"_a)
                .def(py::init(
                         [](py::array_t<int> b)
                         {
                             py::buffer_info info = b.request();

                             // Copy for now, is there a better method?
                             glm::ivec2 v = glm::make_vec2(static_cast<int*>(info.ptr));

                             return v;
                         }),
                     "array"_a)
                .def("numpy",
                     [](glm::ivec2& v) -> py::array_t<int>
                     { return py::array_t<int>({2}, {sizeof(int)}, glm::value_ptr(v)); })
                .def_readwrite("x", &glm::ivec2::x)
                .def_readwrite("y", &glm::ivec2::y);

            m_vec3.def(py::init<float, float, float>(), "x"_a, "y"_a, "z"_a)
                .def(py::init<float>(), "value"_a)
                .def(py::init(
                         [](py::array_t<float> b)
                         {
                             py::buffer_info info = b.request();

                             // Copy for now, is there a better method?
                             glm::vec3 v = glm::make_vec3(static_cast<float*>(info.ptr));

                             return v;
                         }),
                     "array"_a)
                .def("numpy",
                     [](glm::vec3& v) -> py::array_t<float>
                     { return py::array_t<float>({3}, {sizeof(float)}, glm::value_ptr(v)); })
                .def_readwrite("x", &glm::vec3::x)
                .def_readwrite("y", &glm::vec3::y)
                .def_readwrite("z", &glm::vec3::z);

            m_ivec3.def(py::init<int, int, int>(), "x"_a, "y"_a, "z"_a)
                .def(py::init<int>(), "value"_a)
                .def(py::init(
                         [](py::array_t<int> b)
                         {
                             py::buffer_info info = b.request();

                             // Copy for now, is there a better method?
                             glm::ivec3 v = glm::make_vec3(static_cast<int*>(info.ptr));

                             return v;
                         }),
                     "array"_a)
                .def("numpy",
                     [](glm::ivec3& v) -> py::array_t<float>
                     { return py::array_t<int>({3}, {sizeof(int)}, glm::value_ptr(v)); })
                .def_readwrite("x", &glm::ivec3::x)
                .def_readwrite("y", &glm::ivec3::y)
                .def_readwrite("z", &glm::ivec3::z);

            m_u32vec3.def(py::init<uint32_t, uint32_t, uint32_t>(), "x"_a, "y"_a, "z"_a)
                .def(py::init<uint32_t>(), "value"_a)
                .def(py::init(
                         [](py::array_t<uint32_t> b)
                         {
                             py::buffer_info info = b.request();

                             // Copy for now, is there a better method?
                             glm::u32vec3 v = glm::make_vec3(static_cast<uint32_t*>(info.ptr));

                             return v;
                         }),
                     "array"_a)
                .def("numpy",
                     [](glm::u32vec3& v) -> py::array_t<uint32_t>
                     { return py::array_t<uint32_t>({3}, {sizeof(uint32_t)}, glm::value_ptr(v)); })
                .def_readwrite("x", &glm::u32vec3::x)
                .def_readwrite("y", &glm::u32vec3::y)
                .def_readwrite("z", &glm::u32vec3::z);

            m_vec4
                .def(py::init(
                         [](py::array_t<float> b)
                         {
                             py::buffer_info info = b.request();

                             // Copy for now, is there a better method?
                             glm::vec4 v = glm::make_vec4(static_cast<float*>(info.ptr));

                             return v;
                         }),
                     "array"_a)
                .def(py::init<float, float, float, float>(), "x"_a, "y"_a, "z"_a, "w"_a)
                .def(py::init<float>(), "value"_a)
                .def("numpy",
                     [](glm::vec4& v) -> py::array_t<float>
                     { return py::array_t<float>({4}, {sizeof(float)}, glm::value_ptr(v)); })
                .def_readwrite("x", &glm::vec4::x)
                .def_readwrite("y", &glm::vec4::y)
                .def_readwrite("z", &glm::vec4::z)
                .def_readwrite("w", &glm::vec4::w);

            m_ivec4.def(py::init<int, int, int, int>(), "x"_a, "y"_a, "z"_a, "w"_a)
                .def(py::init<int>(), "value"_a)
                .def(py::init(
                         [](py::array_t<int> b)
                         {
                             py::buffer_info info = b.request();

                             // Copy for now, is there a better method?
                             glm::ivec4 v = glm::make_vec4(static_cast<int*>(info.ptr));

                             return v;
                         }),
                     "array"_a)
                .def("numpy",
                     [](glm::ivec4& v) -> py::array_t<int>
                     { return py::array_t<int>({4}, {sizeof(int)}, glm::value_ptr(v)); })
                .def_readwrite("x", &glm::ivec4::x)
                .def_readwrite("y", &glm::ivec4::y)
                .def_readwrite("z", &glm::ivec4::z)
                .def_readwrite("w", &glm::ivec4::w);

            m_mat3
                .def(py::init(
                         [](py::array_t<float, py::array::c_style | py::array::forcecast> b)
                         {
                             py::buffer_info info = b.request();

                             glm::mat3 M;

                             const float* data = static_cast<const float*>(b.data());
                             for(int i = 0; i < 3; ++i)
                             {
                                 for(int j = 0; j < 3; ++j)
                                 {
                                     M[i][j] = data[b.index_at(j, i)];
                                 }
                             }

                             return M;
                         }),
                     "array"_a)
                .def("numpy",
                     [](glm::mat3& M) -> py::array_t<float>
                     {
                         float data[3][3];

                         for(int i = 0; i < 3; ++i)
                         {
                             for(int j = 0; j < 3; ++j)
                             {
                                 data[i][j] = M[j][i];
                             }
                         }

                         return py::array_t<float>({3, 3}, {sizeof(float) * 3, sizeof(float)}, (const float*)data);
                     });

            m_mat4
                .def(py::init(
                         [](py::array_t<float, py::array::c_style | py::array::forcecast> b)
                         {
                             py::buffer_info info = b.request();

                             glm::mat4 M;

                             const float* data = static_cast<const float*>(b.data());
                             for(int i = 0; i < 4; ++i)
                             {
                                 for(int j = 0; j < 4; ++j)
                                 {
                                     M[i][j] = data[b.index_at(j, i)];
                                 }
                             }

                             return M;
                         }),
                     "array"_a)
                .def("numpy",
                     [](glm::mat4& M) -> py::array_t<float>
                     {
                         float data[4][4];

                         for(int i = 0; i < 4; ++i)
                         {
                             for(int j = 0; j < 4; ++j)
                             {
                                 data[i][j] = M[j][i];
                             }
                         }

                         return py::array_t<float>({4, 4}, {sizeof(float) * 4, sizeof(float)}, (const float*)data);
                     });
        });
}
}    // namespace python
}    // namespace atcg
