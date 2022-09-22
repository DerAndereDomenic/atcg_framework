#include <DataStructure/PointCloud.h>

#include <DataStructure/Mesh.h>

namespace atcg
{
    namespace IO
    {
        std::shared_ptr<PointCloud> read_pointcloud(const char* path)
        {
            std::shared_ptr<PointCloud> cloud = std::make_shared<PointCloud>();

            Mesh dummy_mesh;
            OpenMesh::IO::read_mesh(dummy_mesh, path);
            dummy_mesh.request_vertex_colors();
            dummy_mesh.request_face_normals();
            dummy_mesh.update_normals();

            for(auto vertex : dummy_mesh.vertices())
            {
                Mesh::Point p = dummy_mesh.point(vertex);
                Mesh::Normal n = dummy_mesh.calc_vertex_normal(vertex);
                Mesh::Color c = dummy_mesh.color(vertex);
                PointCloud::VertexHandle vh = cloud->add_vertex(p);

                cloud->set_color(vh, c);
                cloud->set_normal(vh, n);
            }

            return cloud;
        }
    }
}