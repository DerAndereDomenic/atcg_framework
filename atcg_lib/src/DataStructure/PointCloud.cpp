#include <DataStructure/PointCloud.h>

#include <DataStructure/Mesh.h>

#include <fstream>
#include <string>
#include <iostream>
#include <vector>

namespace atcg
{
namespace IO
{

namespace detail
{
std::vector<std::string> split_string(const std::string& str, const char delimiter)
{
    std::vector<std::string> res;
    size_t start = 0u;
    size_t end   = str.find(delimiter);
    while(end != std::string::npos)
    {
        res.push_back(str.substr(start, end - start));
        start = end + 1;
        end   = str.find(delimiter, start);
    }

    res.push_back(str.substr(start, end));

    return res;
}

void parse_xyz_file(const atcg::ref_ptr<PointCloud>& cloud, const char* path)
{
    std::string line;
    std::ifstream file;
    file.open(path);

    if(!file.is_open()) return;

    std::getline(file, line);

    bool has_position_data = false, has_color_data = false;
    int32_t x = -1, y = -1, z = -1, r = -1, g = -1, b = -1;
    auto substrings = split_string(line, ' ');

    for(uint32_t i = 0; i < substrings.size(); ++i)
    {
        if(substrings[i] == "X")
            x = i;
        else if(substrings[i] == "Y")    // Convert between xyz and opengl coordinate system -> swap y and z
            y = i;
        else if(substrings[i] == "Z")
            z = i;
        else if(substrings[i] == "R")
            r = i;
        else if(substrings[i] == "G")
            g = i;
        else if(substrings[i] == "B")
            b = i;
    }
    has_position_data = x >= 0 && y >= 0 && z >= 0;
    has_color_data    = r >= 0 && g >= 0 && b >= 0;

    if(!has_position_data)
    {
        std::cerr << "Can only create point cloud that has spatial information!\n";
        return;
    }

    while(std::getline(file, line))
    {
        auto data = split_string(line, ' ');

        float x_ = std::stof(data[x]);
        float y_ = -std::stof(data[y]);
        float z_ = std::stof(data[z]);
        auto vh  = cloud->add_vertex(PointCloud::Point {x_, y_, z_});

        uint8_t r_ = has_color_data ? static_cast<uint8_t>(std::stoi(data[r])) : 255u;
        uint8_t g_ = has_color_data ? static_cast<uint8_t>(std::stoi(data[g])) : 255u;
        uint8_t b_ = has_color_data ? static_cast<uint8_t>(std::stoi(data[b])) : 255u;
        cloud->set_color(vh, PointCloud::Color {r_, g_, b_});
    }
}
}    // namespace detail

atcg::ref_ptr<PointCloud> read_pointcloud(const char* path)
{
    atcg::ref_ptr<PointCloud> cloud = atcg::make_ref<PointCloud>();

    std::string path_str    = path;
    std::string file_ending = path_str.substr(path_str.size() - 4);

    if(file_ending == ".xyz") { detail::parse_xyz_file(cloud, path); }
    else
    {
        Mesh dummy_mesh;
        OpenMesh::IO::read_mesh(dummy_mesh, path);
        dummy_mesh.request_vertex_colors();
        dummy_mesh.request_face_normals();
        dummy_mesh.update_normals();

        for(auto vertex: dummy_mesh.vertices())
        {
            Mesh::Point p               = dummy_mesh.point(vertex);
            Mesh::Normal n              = dummy_mesh.calc_vertex_normal(vertex);
            Mesh::Color c               = dummy_mesh.color(vertex);
            PointCloud::VertexHandle vh = cloud->add_vertex(p);

            cloud->set_color(vh, c);
            cloud->set_normal(vh, n);
        }
    }

    return cloud;
}
}    // namespace IO
}    // namespace atcg