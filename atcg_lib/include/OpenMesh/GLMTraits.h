#pragma once

#include <OpenMesh/Core/Mesh/Traits.hh>
#include <glm/glm.hpp>

namespace OpenMesh
{
template<typename _Scalar, int _Dimension>
struct vector_traits<glm::vec<_Dimension, _Scalar>>
{
    using vector_type         = glm::vec<_Dimension, _Scalar>;
    using value_type          = _Scalar;
    static const size_t size_ = _Dimension;
    static size_t size() { return size_; }
};
}    // namespace OpenMesh

namespace glm
{

template<typename Scalar, int Dimension>
glm::vec<Dimension, Scalar> cross(const glm::vec<Dimension, Scalar>& x, const glm::vec<Dimension, Scalar>&& y)
{
    return glm::cross(x, y);
}

template<typename Scalar, int Dimension>
Scalar norm(const glm::vec<Dimension, Scalar>& x)
{
    return glm::length(x);
}

template<typename Scalar, int Dimension>
Scalar sqrnorm(const glm::vec<Dimension, Scalar>& x)
{
    return glm::dot(x, x);
}

template<typename Scalar, int Dimension>
glm::vec<Dimension, Scalar> vectorize(glm::vec<Dimension, Scalar>& x, Scalar const& val)
{
    x = glm::vec<Dimension, Scalar>(val);
    return x;
}

}    // namespace glm

struct GLMTraits : OpenMesh::DefaultTraits
{
    /// The default coordinate type is OpenMesh::Vec3f.
    typedef glm::vec3 Point;

    /// The default normal type is OpenMesh::Vec3f.
    typedef glm::vec3 Normal;

    /// The default 2D texture coordinate type is OpenMesh::Vec2f.
    typedef glm::vec2 TexCoord2D;
    /// The default 3D texture coordinate type is OpenMesh::Vec3f.
    typedef glm::vec3 TexCoord3D;

    /// The default color type is OpenMesh::Vec3uc.
    typedef glm::vec3 Color;
};