#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
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

    void marching_cubes(const std::shared_ptr<SDFGrid>& grid, const std::shared_ptr<atcg::Mesh>& mesh)
    {
        mesh->request_vertex_colors();
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
            atcg::Mesh::VertexHandle v_handles[12];

            atcg::Mesh::Color clr;
            clr[0] = 255;
            clr[1] = 0;
            clr[2] = 0;

            #define CREATE_VERTEX_ON_EDGE(n, corner_i, corner_j) \
                if(edge_table[cubeindex] & (1 << n)) \
                {\
                    vertex_list[n] = interpolate_point(ISOVALUE, voxel_positions[corner_i], sdf_values[corner_i], voxel_positions[corner_j], sdf_values[corner_j]);\
                    v_handles[n] = mesh->add_vertex(atcg::Mesh::Point(vertex_list[n].x, vertex_list[n].y, vertex_list[n].z)); \
                    mesh->set_color(v_handles[n], atcg::Mesh::Color(clr)); \
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
                atcg::Mesh::VertexHandle face_vhandles[3];

                face_vhandles[0] = v_handles[triangle_table[cubeindex][i + 0]];
                face_vhandles[1] = v_handles[triangle_table[cubeindex][i + 1]];
                face_vhandles[2] = v_handles[triangle_table[cubeindex][i + 2]];
                mesh->add_face(face_vhandles[0], face_vhandles[1], face_vhandles[2]);
            }
        }
    }

    void subdivide_mesh(const std::shared_ptr<atcg::Mesh>& mesh)
    {
        //TODO: sqrt(3) subdivision
        mesh->request_edge_status();
        mesh->request_vertex_status();
        uint32_t num_features = mesh->find_feature_edges();
        std::cout << "Found " << num_features << " feature edges\n";
        
        uint32_t nv = mesh->n_vertices();
        uint32_t ne = mesh->n_edges();
        uint32_t nf = mesh->n_faces();

        auto eend = mesh->edges_end();
        auto fend = mesh->faces_end();

        auto new_pos_property = OpenMesh::makeTemporaryProperty<OpenMesh::VertexHandle, atcg::Mesh::Point>(*mesh.get());

        //Calculate new position of old vertices
        for(auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end(); ++v_it)
        {
            uint32_t n = (*v_it).valence();
            float beta = (4.0f - 2.0f * cosf(2.0f * static_cast<float>(M_PI) / static_cast<float>(n))) / 9.0f;
            new_pos_property[*v_it] = {0,0,0};

            for(auto vv_it = v_it->vertices().begin(); vv_it != v_it->vertices().end(); ++vv_it)
            {
                new_pos_property[*v_it] += mesh->point(*vv_it);
            }

            new_pos_property[*v_it] = (1.0f - beta) * mesh->point(*v_it) + beta/static_cast<float>(n) * new_pos_property[*v_it];

            if(mesh->is_boundary(*v_it) || v_it->feature())
            {
                new_pos_property[*v_it] = mesh->point(*v_it);
            }
        }

        //Split faces
        std::vector<atcg::Mesh::FaceHandle> faces;
        std::vector<atcg::Mesh::VertexHandle> centroids;
        for(auto f_it = mesh->faces_begin(); f_it != fend; ++f_it)
        {
            atcg::Mesh::Point center = {0,0,0};
            for(auto v_it = f_it->vertices().begin(); v_it != f_it->vertices().end(); ++v_it)
            {
                center += mesh->point(*v_it);
            }

            center /= 3.0f;
            auto handle = mesh->new_vertex(center);
            new_pos_property[handle] = center;
            mesh->split(*f_it, handle);
        }

        //Set new vertex positions
        for(auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end(); ++v_it)
        {
            if(mesh->is_boundary(*v_it)  || v_it->feature())continue;
            mesh->point(*v_it) = new_pos_property[*v_it];
        }

        //Flip old edges
        for(auto e_it = mesh->edges_begin(); e_it != eend; ++e_it)
        {
            atcg::Mesh::EdgeHandle e = *e_it;

            if(!mesh->is_flip_ok(e) || e_it->feature()) continue;
            mesh->flip(e);
        }

        //Update rendering
        mesh->uploadData();
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {

        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        //Create a grid at the origin with 10 voxels per side length with a size of 0.1
        grid = std::make_shared<SDFGrid>(glm::vec3(0), 70, 0.05f);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        if(mesh && render_faces)
            atcg::Renderer::draw(mesh, atcg::ShaderManager::getShader("base"), camera_controller->getCamera());

        if(mesh && render_points)
            atcg::Renderer::drawPoints(mesh, glm::vec3(0), atcg::ShaderManager::getShader("flat"), camera_controller->getCamera());

        if(mesh && render_edges)
            atcg::Renderer::drawLines(mesh, glm::vec3(0), camera_controller->getCamera());

        if(render_grid)
            atcg::Renderer::drawGrid(grid->getGridDimensions(), camera_controller->getCamera(), update_grid);
    }

    virtual void onImGuiRender() override
    {
        update_grid = false;
        ImGui::BeginMainMenuBar();

        if(ImGui::BeginMenu("Rendering"))
        {
            ImGui::MenuItem("Show Render Settings", nullptr, &show_render_settings);
            ImGui::EndMenu();
        }

        if(ImGui::BeginMenu("Exercise"))
        {
            ImGui::MenuItem("Marching Cubes", nullptr, &show_marching_cubes);
            ImGui::MenuItem("Subdivision", nullptr, &show_subdivision);
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();

        if(show_marching_cubes)
        {
            ImGui::Begin("Settings MC", &show_marching_cubes);
            if(ImGui::SliderFloat("Voxel Size", &voxel_size, 0.01f, 0.2f))
            {
                uint32_t num_voxels = static_cast<uint32_t>(70.0f/voxel_size * 0.05f);
                num_voxels += (1 - num_voxels%2);
                grid = std::make_shared<SDFGrid>(glm::vec3(0), static_cast<uint32_t>(70.0f/voxel_size * 0.05f) , voxel_size);
                update_grid = true;
            }

            if(ImGui::Button("Run"))
            {
                //Fill the grid with the sdf
                mesh = std::make_shared<atcg::Mesh>();
                fillGrid(grid);

                marching_cubes(grid, mesh);

                render_grid = false;

                mesh->uploadData();
            }

            ImGui::End();
        }

        if(show_subdivision)
        {
            ImGui::Begin("Settings SD", &show_subdivision);

            if(ImGui::Button("Subdivide"))
            {
                subdivide_mesh(mesh);
            }

            ImGui::End();
        }

        if(show_render_settings)
        {
            ImGui::Begin("Settings", &show_render_settings);

            ImGui::Checkbox("Render Vertices", &render_points);
            ImGui::Checkbox("Render Edges", &render_edges);
            ImGui::Checkbox("Render Mesh", &render_faces);
            ImGui::Checkbox("Render Grid", &render_grid);
            ImGui::End();
        }


    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event& event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
        dispatcher.dispatch<atcg::FileDroppedEvent>(ATCG_BIND_EVENT_FN(G02Layer::onFileDropped));
    }

    bool onFileDropped(atcg::FileDroppedEvent& event)
    {
        mesh = atcg::IO::read_mesh(event.getPath().c_str());

        mesh->uploadData();

        //Also reset camera
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        return true;
    }

private:
    std::shared_ptr<atcg::Mesh> mesh;
    std::shared_ptr<atcg::CameraController> camera_controller;
    std::shared_ptr<SDFGrid> grid;

    bool show_marching_cubes = true;
    bool show_render_settings = false;
    bool render_faces = true;
    bool render_points = false;
    bool render_edges = false;
    bool update_grid = false;
    bool render_grid = true;
    float voxel_size = 0.05f;

    bool show_subdivision = true;
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