#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <glfw/glfw3.h>
#include <imgui.h>
#include <algorithm>

#include "MarchingCubesTable.h"

struct SDFVoxel
{
    float sdf;
};

using SDFGrid = atcg::Grid<SDFVoxel>;

class G02Layer : public atcg::Layer
{
public:

    G02Layer(const std::string& name) : atcg::Layer(name) {}

    float sdf_heart(const glm::vec3& p)
    {
        float x = p.x;
        float y = p.y;
        float z = p.z;
        return std::pow(x * x + 9. / 4. * y * y + z * z - 1, 3) - x * x * z * z * z - 9. / 80. * y * y * z * z * z; 
    }

    void fillGrid(const std::shared_ptr<SDFGrid>& grid)
    {
        for(uint32_t i = 0; i < grid->voxels_per_volume(); ++i)
        {
            glm::ivec3 voxel = grid->index2voxel(i);
            glm::vec3 p = grid->voxel2position(voxel);

            (*grid)[i].sdf = sdf_heart(p);
        }
    }

    inline glm::vec3 interpolate_point(const float isovalue,
                                  const glm::vec3 vector_1,
                                  const float sdf_1,
                                  const glm::vec3 vector_2,
                                  const float sdf_2)
    {
        if(std::abs(sdf_1 - sdf_2) < 1e-8f)
            return vector_1;
        
        float t = (isovalue - sdf_1) / (sdf_2 - sdf_1);
        return vector_1 + t * (vector_2 - vector_1);
        //return (vector_1 + vector_2)/2.0f;
    }

    void marching_cubes(const std::shared_ptr<SDFGrid>& grid, const std::shared_ptr<atcg::TriMesh>& mesh)
    {
        for(uint32_t index = 0; index < grid->voxels_per_volume(); ++index)
        {
            glm::ivec3 voxel = grid->index2voxel(index);
            glm::vec3 voxel_start = grid->voxel2position(voxel);

            const float I = grid->voxel_side_length();
            const float O = 0.0f;

            //Get neighboring voxel positions
            glm::vec3 voxel_positions[8] = {
                glm::vec3(voxel_start.x + O, voxel_start.y + O, voxel_start.z + O),
                glm::vec3(voxel_start.x + I, voxel_start.y + O, voxel_start.z + O),
                glm::vec3(voxel_start.x + I, voxel_start.y + I, voxel_start.z + O),
                glm::vec3(voxel_start.x + O, voxel_start.y + I, voxel_start.z + O),
                glm::vec3(voxel_start.x + O, voxel_start.y + O, voxel_start.z + I),
                glm::vec3(voxel_start.x + I, voxel_start.y + O, voxel_start.z + I),
                glm::vec3(voxel_start.x + I, voxel_start.y + I, voxel_start.z + I),
                glm::vec3(voxel_start.x + O, voxel_start.y + I, voxel_start.z + I),
            };

            //Get voxel information
            float sdf_values[8];
            bool all_valid = true;
            for(uint32_t i = 0; i < 8; ++i)
            {
                //Abort if a neighbor is missing (at the broder of volume)
                if(!grid->insideVolume(voxel_positions[i]))
                {
                    all_valid = false;
                    break;        
                }

                sdf_values[i] = (*grid)(voxel_positions[i]).sdf;
            }

            if(!all_valid)
                continue;

            const float ISOVALUE = 0.0f;

            #define SET_BIT(ind, i) \
                ind |= (1 << i)

            uint8_t cubeindex = 0;
            for(uint32_t i = 0; i < 8; ++i)
            {
                if(sdf_values[i] >= ISOVALUE)
                {
                    SET_BIT(cubeindex, i);
                }
            }

            #undef SET_BIT

            //No edges -> early out
            if(edge_table[cubeindex] == 0)
                continue;

            glm::vec3 vertex_list[12];
            atcg::TriMesh::VertexHandle v_handles[12];

            #define CREATE_VERTEX_ON_EDGE(n, corner_i, corner_j) \
                if(edge_table[cubeindex] & (1 << n)) \
                {\
                    vertex_list[n] = interpolate_point(ISOVALUE, voxel_positions[corner_i], sdf_values[corner_i], voxel_positions[corner_j], sdf_values[corner_j]);\
                    v_handles[n] = mesh->add_vertex(atcg::TriMesh::Point(vertex_list[n].x, vertex_list[n].y, vertex_list[n].z)); \
                }

            CREATE_VERTEX_ON_EDGE( 0, 0, 1);
            CREATE_VERTEX_ON_EDGE( 1, 1, 2);
            CREATE_VERTEX_ON_EDGE( 2, 2, 3);
            CREATE_VERTEX_ON_EDGE( 3, 3, 0);
            CREATE_VERTEX_ON_EDGE( 4, 4, 5);
            CREATE_VERTEX_ON_EDGE( 5, 5, 6);
            CREATE_VERTEX_ON_EDGE( 6, 6, 7);
            CREATE_VERTEX_ON_EDGE( 7, 7, 4);
            CREATE_VERTEX_ON_EDGE( 8, 0, 4);
            CREATE_VERTEX_ON_EDGE( 9, 1, 5);
            CREATE_VERTEX_ON_EDGE(10, 2, 6);
            CREATE_VERTEX_ON_EDGE(11, 3, 7);

            #undef CREATE_VERTEX_ON_EDGE

            for(uint32_t i = 0; triangle_table[cubeindex][i] != -1; i += 3)
            {
                atcg::TriMesh::VertexHandle face_vhandles[3];

                face_vhandles[0] = v_handles[triangle_table[cubeindex][i + 0]];
                face_vhandles[1] = v_handles[triangle_table[cubeindex][i + 1]];
                face_vhandles[2] = v_handles[triangle_table[cubeindex][i + 2]];
                mesh->add_face(face_vhandles[0], face_vhandles[1], face_vhandles[2]);
            }
        }
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        mesh = std::make_shared<atcg::TriMesh>();

        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        //Create a grid at the origin with 10 voxels per side length with a size of 0.1
        grid = std::make_shared<SDFGrid>(glm::vec3(0), 100, 0.05f);

        //Fill the grid with the sdf
        fillGrid(grid);

        marching_cubes(grid, mesh);

        render_mesh = std::make_shared<atcg::RenderMesh>();
        render_mesh->uploadData(mesh);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        atcg::Renderer::draw(render_mesh, atcg::ShaderManager::getShader("base"), camera_controller->getCamera());
    }

    virtual void onImGuiRender() override
    {
        ImGui::BeginMainMenuBar();

        if(ImGui::BeginMenu("Exercise"))
        {
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();

    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event& event) override
    {
        camera_controller->onEvent(event);
    }

private:
    std::shared_ptr<atcg::TriMesh> mesh;
    std::shared_ptr<atcg::RenderMesh> render_mesh;
    std::shared_ptr<atcg::CameraController> camera_controller;
    std::shared_ptr<SDFGrid> grid;
};

class G02 : public atcg::Application
{
    public:

    G02()
        :atcg::Application()
    {
        pushLayer(new G02Layer("Layer"));
    }

    ~G02() {}

};

atcg::Application* atcg::createApplication()
{
    return new G02;
}