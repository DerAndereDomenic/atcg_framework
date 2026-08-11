#pragma once

#include <Core/glm.h>
#include <Core/Platform.h>
#include <Core/CUDA.h>

namespace atcg
{

/**
 * @brief A local coordinate frame defined by three orthonormal vectors. Provides functions to convert between local and
 * world coordinates.
 */
template<typename T>
struct Frame
{
    /**
     * @brief Default constructor. Leaves the content of the frame uninitialized. The user should set the frame using
     * other constructors or by directly writing to the localX/localY/localZ members.
     */
    Frame() = default;

    /**
     * @brief Construct a frame with the given local X, Y and Z directions. The user should ensure that the three
     * vectors are orthonormal.
     *
     * @param localX The local X direction.
     * @param localY The local Y direction.
     * @param localZ The local Z direction.
     */
    ATCG_HOST_DEVICE Frame(const T& localX, const T& localY, const T& localZ)
        : _localX(localX),
          _localY(localY),
          _localZ(localZ)
    {
    }

    /**
     * @brief Construct a frame with the given local Z direction. The local X and Y directions are computed using an
     * arbitrary but consistent method to ensure that the three vectors are orthonormal.
     *
     * @param localZ The local Z direction.
     */
    ATCG_HOST_DEVICE Frame(const T& localZ)
    {
        float x  = localZ.x;
        float y  = localZ.y;
        float z  = localZ.z;
        float sz = (z >= 0) ? 1 : -1;
        float a  = 1 / (sz + z);
        float ya = y * a;
        float b  = x * ya;
        float c  = x * sz;

        _localX = T(c * x * a - 1, sz * b, c);
        _localY = T(b, y * ya - sz, y);
        _localZ = localZ;
    }

    /**
     * @brief Convert a vector from local coordinates to world coordinates using the frame's orthonormal basis. The
     * input vector is expressed in the local coordinate system defined by the frame, and the output vector is expressed
     * in the world coordinate system.
     *
     * @param local The vector in local coordinates to be transformed to world coordinates.
     * @return The transformed vector in world coordinates.
     */
    ATCG_HOST_DEVICE auto toWorld(const T& local) const
    {
        return local.x * _localX + local.y * _localY + local.z * _localZ;
    }

    /**
     * @brief Convert a vector from world coordinates to local coordinates using the frame's orthonormal basis. The
     * input vector is expressed in the world coordinate system, and the output vector is expressed in the local
     * coordinate system defined by the frame. The conversion is done by projecting the world vector onto the local axes
     * of the frame.
     *
     * @param world The vector in world coordinates to be transformed to local coordinates.
     * @return The transformed vector in local coordinates.
     */
    ATCG_HOST_DEVICE auto toLocal(const T& world) const
    {
        return T(glm::dot(world, _localX), glm::dot(world, _localY), glm::dot(world, _localZ));
    }

    /**
     * @brief Compute the cosine of the angle between the local Z direction of the frame and a given local vector. This
     * is equivalent to the Z component of the input vector when expressed in the local coordinate system of the frame.
     * The input vector is expected to be normalized and expressed in local coordinates.
     *
     * @param local The vector in local coordinates for which to compute the cosine of the angle with the local Z
     * direction.
     *
     * @return The cosine of the angle between the local Z direction and the input vector, which is the Z component of
     * the input vector in local coordinates.
     */
    ATCG_HOST_DEVICE auto cosTheta(const T& local) const { return local.z; }

    /**
     * @brief Compute the sine of the angle between the local Z direction of the frame and a given local vector. This is
     * equivalent to the length of the projection of the input vector onto the local XY plane of the frame. The input
     * vector is expected to be normalized and expressed in local coordinates.
     *
     * @param local The vector in local coordinates for which to compute the sine of the angle with the local Z
     * direction.
     *
     * @return The sine of the angle between the local Z direction and the input vector, which is the length of the
     * projection of the input vector onto the local XY plane of the frame.
     */
    ATCG_HOST_DEVICE auto sinTheta(const T& local) const { return glm::sqrt(glm::max(T(0), T(1) - local.z * local.z)); }

    /**
     * @brief Accessor for the local X direction of the frame. Returns a reference to the local X vector, which is one
     * of the three orthonormal basis vectors that define the frame.
     *
     * @return A reference to the local X vector of the frame.
     */
    ATCG_HOST_DEVICE const T& localX() const { return _localX; }

    /**
     * @brief Accessor for the local Y direction of the frame. Returns a reference to the local Y vector, which is one
     * of the three orthonormal basis vectors that define the frame.
     *
     * @return A reference to the local Y vector of the frame.
     */
    ATCG_HOST_DEVICE const T& localY() const { return _localY; }

    /**
     * @brief Accessor for the local Z direction of the frame. Returns a reference to the local Z vector, which is one
     * of the three orthonormal basis vectors that define the frame.
     *
     * @return A reference to the local Z vector of the frame.
     */
    ATCG_HOST_DEVICE const T& localZ() const { return _localZ; }

private:
    T _localX;
    T _localY;
    T _localZ;
};
}    // namespace atcg