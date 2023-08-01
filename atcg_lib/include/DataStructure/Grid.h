#pragma once

#include <Core/glm.h>

#include <string>

namespace atcg
{
struct GridDimension
{
    glm::vec3 origin    = glm::vec3(0);
    uint32_t num_voxels = 0;
    float voxel_length  = 0.0f;
};

template<class VoxelT>
class Grid
{
public:
    Grid() = default;

    /**
     * @brief Construct a new Grid object.
     * This creates an "empty" grid that only stores information about voxel sizes and arangement
     *
     * @param origin The origin of the grid
     * @param num_voxels The number of voxels in each direction
     * @param voxel_length The side length of a voxel
     * @param allocate If the memory should be allocated if we only want to store the spatial information
     */
    Grid(const glm::vec3& origin, const uint32_t& num_voxels, const float& voxel_length, bool allocate = true);

    /**
     * @brief Destroy the Grid object
     */
    ~Grid();

    /**
     * @brief  Get the side length of the volume in number of voxels
     * @return The number of voxels in each side
     */
    uint32_t num_voxels() const;

    /**
     * @brief The number of voxels in the volume
     * @return The number of voxels in the volume
     */
    uint32_t voxels_per_volume() const;

    /**
     * @brief Get the side length of a single voxel
     * @return Voxel side length
     */
    float voxel_side_length() const;

    /**
     * @brief Get the grid origin
     * @return The origin
     */
    glm::vec3 origin() const;

    /**
     * @brief Get the voxel according to the index inside the volume
     * @param index The index of the voxel
     * @return Reference to the voxel
     */
    VoxelT& operator[](const uint32_t& index);

    /**
     * @brief Get the voxel according to the 3D position
     * @param position The 3D position
     * @return Reference to the voxel
     */
    VoxelT& operator()(const glm::vec3& position);

    template<class SelectionFunctor>
    typename SelectionFunctor::value_type readInterpolated(const glm::vec3& position, const SelectionFunctor& selector);

    /**
     * @brief Get the voxel corresponding to a given index
     * @param index The index
     * @return The 3D grid position
     */
    glm::ivec3 index2voxel(const int32_t& index);

    /**
     * @brief Get the index of a specific voxel
     * @param voxel The voxel
     * @return The corresponding index
     */
    int32_t voxel2index(const glm::ivec3& voxel);

    /**
     * @brief Compute the 3D position associated with a voxel (center)
     * @param voxel The voxel
     * @return The 3D position of the voxel center
     */
    glm::vec3 voxel2position(const glm::ivec3& voxel);

    /**
     * @brief Compute the voxel coordinates from a given point
     * @param position The 3D position
     * @return The grid coordinates
     */
    glm::ivec3 position2voxel(const glm::vec3& position);

    /**
     * @brief Get the voxel center of the current position
     * @param position The position
     * @return The voxel center
     */
    glm::vec3 voxel_center(const glm::vec3& position);

    /**
     * @brief Check if a point is inside the volume
     * @param position The position to check
     * @return True if the point is inside the bounding volume
     */
    bool insideVolume(const glm::vec3& position);

    /**
     * @brief Get the internal data pointer
     * @return Pointer to the start of the data
     */
    VoxelT* data();

    /**
     * @brief Copy data into the grid
     *
     * @param data The data to copy
     */
    void setData(VoxelT* data);

    /**
     * @brief Get positional information about the grid
     *
     * @return The grid dimension
     */
    inline GridDimension getGridDimensions() const { return _dim; }

private:
    GridDimension _dim  = {};
    VoxelT* _voxel_pool = nullptr;
};


///
/// IMPLEMENTATION
///
template<class VoxelT>
Grid<VoxelT>::Grid(const glm::vec3& origin, const uint32_t& num_voxels, const float& voxel_length, bool allocate)
    : _dim({origin, num_voxels, voxel_length})
{
    if(allocate) _voxel_pool = new VoxelT[num_voxels * num_voxels * num_voxels];
}

template<class VoxelT>
Grid<VoxelT>::~Grid()
{
    delete[] _voxel_pool;
    _voxel_pool = nullptr;
}

template<class VoxelT>
uint32_t Grid<VoxelT>::num_voxels() const
{
    return _dim.num_voxels;
}

template<class VoxelT>
uint32_t Grid<VoxelT>::voxels_per_volume() const
{
    return _dim.num_voxels * _dim.num_voxels * _dim.num_voxels;
}

template<class VoxelT>
float Grid<VoxelT>::voxel_side_length() const
{
    return _dim.voxel_length;
}

template<class VoxelT>
glm::vec3 Grid<VoxelT>::origin() const
{
    return _dim.origin;
}

template<class VoxelT>
VoxelT& Grid<VoxelT>::operator[](const uint32_t& index)
{
    return _voxel_pool[index];
}

template<class VoxelT>
VoxelT& Grid<VoxelT>::operator()(const glm::vec3& position)
{
    return _voxel_pool[voxel2index(position2voxel(position))];
}

template<class VoxelT>
template<class SelectionFunctor>
typename SelectionFunctor::value_type Grid<VoxelT>::readInterpolated(const glm::vec3& position,
                                                                     const SelectionFunctor& selector)
{
    const float O = -0.5f * _dim.voxel_length;
    const float I = -O;

    glm::vec3 lookups[8] = {
        glm::vec3(position.x + O, position.y + O, position.z + O),    // A 0
        glm::vec3(position.x + O, position.y + O, position.z + I),    // B 1
        glm::vec3(position.x + O, position.y + I, position.z + O),    // C 2
        glm::vec3(position.x + O, position.y + I, position.z + I),    // D 3
        glm::vec3(position.x + I, position.y + O, position.z + O),    // E 4
        glm::vec3(position.x + I, position.y + O, position.z + I),    // F 5
        glm::vec3(position.x + I, position.y + I, position.z + O),    // G 6
        glm::vec3(position.x + I, position.y + I, position.z + I),    // H 7
    };

    typename SelectionFunctor::value_type values[8];
    for(uint32_t i = 0; i < 8; ++i)
    {
        if(insideVolume(lookups[i]))
            values[i] = selector.select(operator()(lookups[i]));
        else
            return std::numeric_limits<typename SelectionFunctor::value_type>::max();
    }

    glm::vec3 corner_voxel = voxel_center(lookups[0]);

    glm::vec3 deltaXYZ = (position - corner_voxel) / _dim.voxel_length;

    typename SelectionFunctor::value_type cae = glm::lerp(values[0], values[4], deltaXYZ.x);
    typename SelectionFunctor::value_type cbf = glm::lerp(values[1], values[5], deltaXYZ.x);
    typename SelectionFunctor::value_type ccg = glm::lerp(values[2], values[6], deltaXYZ.x);
    typename SelectionFunctor::value_type cdh = glm::lerp(values[3], values[7], deltaXYZ.x);

    typename SelectionFunctor::value_type ccae = glm::lerp(cae, cbf, deltaXYZ.y);
    typename SelectionFunctor::value_type ccbf = glm::lerp(ccg, cdh, deltaXYZ.y);

    return glm::lerp(ccae, ccbf, deltaXYZ.z);
}

template<class VoxelT>
glm::ivec3 Grid<VoxelT>::index2voxel(const int32_t& index)
{
    int32_t x = index % _dim.num_voxels;
    int32_t y = (index / _dim.num_voxels) % _dim.num_voxels;
    int32_t z = index / (_dim.num_voxels * _dim.num_voxels);

    return glm::ivec3(x, y, z);
}

template<class VoxelT>
int32_t Grid<VoxelT>::voxel2index(const glm::ivec3& voxel)
{
    int32_t x = voxel.x;
    int32_t y = voxel.y;
    int32_t z = voxel.z;

    return x + y * _dim.num_voxels + z * _dim.num_voxels * _dim.num_voxels;
}

template<class VoxelT>
glm::vec3 Grid<VoxelT>::voxel2position(const glm::ivec3& voxel)
{
    glm::vec3 center = (glm::vec3(voxel) - ((float)_dim.num_voxels - 1.0f) / 2.0f) * _dim.voxel_length;
    return center + _dim.origin;
}

template<class VoxelT>
glm::ivec3 Grid<VoxelT>::position2voxel(const glm::vec3& position)
{
    glm::vec3 center = position - _dim.origin;

    int32_t x = static_cast<int32_t>(center.x / _dim.voxel_length + (float)_dim.num_voxels / 2.0f);
    int32_t y = static_cast<int32_t>(center.y / _dim.voxel_length + (float)_dim.num_voxels / 2.0f);
    int32_t z = static_cast<int32_t>(center.z / _dim.voxel_length + (float)_dim.num_voxels / 2.0f);

    return glm::ivec3(x, y, z);
}

template<class VoxelT>
glm::vec3 Grid<VoxelT>::voxel_center(const glm::vec3& position)
{
    glm::ivec3 voxel = position2voxel(position);
    return (glm::vec3(voxel) - ((float)_dim.num_voxels - 1) / 2.0f) * _dim.voxel_length + _dim.origin;
}

template<class VoxelT>
bool Grid<VoxelT>::insideVolume(const glm::vec3& position)
{
    glm::ivec3 voxel = position2voxel(position);

    return voxel.x >= 0 && voxel.x < _dim.num_voxels && voxel.y >= 0 && voxel.y < _dim.num_voxels && voxel.z >= 0 &&
           voxel.z < _dim.num_voxels;
}

template<class VoxelT>
VoxelT* Grid<VoxelT>::data()
{
    return _voxel_pool;
}

template<class VoxelT>
void Grid<VoxelT>::setData(VoxelT* data)
{
    memcpy(_voxel_pool, data, sizeof(VoxelT) * voxels_per_volume());
}
}    // namespace atcg