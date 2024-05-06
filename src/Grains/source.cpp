#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <ImGuizmo.h>

#include <random>

#include <nanort.h>

#include <glm/gtc/random.hpp>

namespace detail
{
glm::ivec3 position2index(const glm::vec3& x0, const float& voxel_size, const uint32_t& num_cells)
{
    return glm::ivec3(static_cast<int32_t>((x0.x / voxel_size + 0.5f) * num_cells),
                      static_cast<int32_t>((x0.y / voxel_size + 0.5f) * num_cells),
                      static_cast<int32_t>((x0.z / voxel_size + 0.5f) * num_cells));
}

void divmod(const int32_t& a, const int32_t& b, int32_t& div, int32_t& mod)
{
    mod = (a % b + b) % b;
    div = (a - mod) / b;
}

bool check_boundary(const float& sample, const float& voxel_size, const float& radius, float& extra_sample)
{
    if(sample > voxel_size / 2.0f - radius)
    {
        extra_sample = sample - voxel_size;
        return true;
    }
    else if(sample < radius - voxel_size / 2.0f)
    {
        extra_sample = sample + voxel_size;
        return true;
    }
    else
    {
        extra_sample = sample;
        return false;
    }
}

int32_t cell2index(const glm::ivec3& idx, const int32_t& num_cells)
{
    return idx.x + idx.y * num_cells + idx.z * num_cells * num_cells;
}

glm::vec3 sample_rotation(std::mt19937& gen, std::uniform_real_distribution<float>& distrib)
{
    float angle = (1.0f - distrib(gen)) * 2.0f * glm::pi<float>();    // xi in (0,1]
    float z     = distrib(gen) * 2.0f - 1.0f;
    float phi   = distrib(gen) * 2.0f * glm::pi<float>();

    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float x = r * cosf(phi);
    float y = r * sinf(phi);

    glm::vec3 axis = glm::vec3(x, y, z);

    return angle * axis;
}
}    // namespace detail

class GrainLayer : public atcg::Layer
{
public:
    GrainLayer(const std::string& name) : atcg::Layer(name) {}

    void poisson_sample()
    {
        std::cout << "Sampling grains" << std::endl;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distrib(0.0f, 1.0f);

        uint32_t max_tries = 30;
        float radius       = m_grain_size * 2.0f;
        float cell_size    = radius / sqrtf(3.0f);    // 3 dimensional
        int32_t num_cells  = static_cast<int32_t>(m_voxel_length / cell_size);

        int32_t* grid = new int32_t[num_cells * num_cells * num_cells];    //[num_cells][num_cells][num_cells];
        memset(grid, -1, sizeof(int32_t) * num_cells * num_cells * num_cells);
        glm::vec3 x0     = glm::vec3(distrib(gen) * m_voxel_length - m_voxel_length / 2.0f,
                                 distrib(gen) * m_voxel_length - m_voxel_length / 2.0f,
                                 distrib(gen) * m_voxel_length - m_voxel_length / 2.0f);
        glm::ivec3 index = detail::position2index(x0, m_voxel_length, num_cells);

        grid[detail::cell2index(index, num_cells)] = 0;
        std::vector<int32_t> active_list           = {0};
        m_positions.push_back(x0);

        glm::vec3 rotation = detail::sample_rotation(gen, distrib);
        m_rotations.push_back(rotation);

        std::vector<glm::vec3> boundary_spheres;    // A list of spheres that intersect the boundary
        std::vector<glm::vec3> boundary_rotations;

        bool hitx, hity, hitz = false;
        glm::vec3 boundary_sample;
        hitx = detail::check_boundary(x0.x, m_voxel_length, radius / 2.0f, boundary_sample.x);
        hity = detail::check_boundary(x0.y, m_voxel_length, radius / 2.0f, boundary_sample.y);
        hitz = detail::check_boundary(x0.z, m_voxel_length, radius / 2.0f, boundary_sample.z);
        for(uint32_t p = 0; p <= hitx; ++p)
        {
            for(uint32_t q = 0; q <= hity; ++q)
            {
                for(uint32_t l = 0; l <= hitz; ++l)
                {
                    // Don't add default sample
                    if(p == 0 && q == 0 && l == 0) continue;
                    glm::vec3 new_sample = glm::vec3(p == 1 ? boundary_sample.x : x0.x,
                                                     q == 1 ? boundary_sample.y : x0.y,
                                                     l == 1 ? boundary_sample.z : x0.z);
                    boundary_spheres.push_back(new_sample);
                    boundary_rotations.push_back(rotation);
                }
            }
        }

        int32_t ceil = 2;    // ceil(sqrt(dimensions)) = 2
        while(!active_list.empty())
        {
            int32_t active_index = static_cast<int32_t>(distrib(gen) * active_list.size());
            int32_t i            = active_list[active_index];

            glm::vec3 xi  = m_positions[i];
            bool inserted = false;
            for(int32_t k = 0; k < max_tries; ++k)
            {
                // Rejection sample a valid point inside the voxel and uniformly between r and 2r
                glm::vec3 sample = glm::vec3(INFINITY);
                float norm       = 0.0f;
                while(fabsf(sample.x) > m_voxel_length / 2.0f || fabsf(sample.y) > m_voxel_length / 2.0f ||
                      fabsf(sample.z) > m_voxel_length / 2.0f || norm < radius || norm >= 2.0f * radius)
                {
                    glm::vec3 box = glm::vec3(2.0f * distrib(gen) * 2.0f * radius - 2.0f * radius,
                                              2.0f * distrib(gen) * 2.0f * radius - 2.0f * radius,
                                              2.0f * distrib(gen) * 2.0f * radius - 2.0f * radius);
                    norm          = length(box);
                    sample        = xi + box;
                }

                glm::ivec3 sample_index = detail::position2index(sample, m_voxel_length, num_cells);

                bool valid = true;
                for(int32_t p = -ceil; p < ceil + 1; ++p)
                {
                    for(int32_t q = -ceil; q < ceil + 1; ++q)
                    {
                        for(int32_t l = -ceil; l < ceil + 1; ++l)
                        {
                            glm::ivec3 temp = sample_index + glm::ivec3(p, q, l);
                            glm::ivec3 div;
                            detail::divmod(temp.x, num_cells, div.x, index.x);
                            detail::divmod(temp.y, num_cells, div.y, index.y);
                            detail::divmod(temp.z, num_cells, div.z, index.z);

                            if(grid[detail::cell2index(index, num_cells)] != -1)
                            {
                                glm::vec3 neighbor = m_positions[grid[detail::cell2index(index, num_cells)]];
                                if(length(neighbor + glm::vec3(div) * m_voxel_length - sample) < radius) valid = false;
                            }
                        }
                    }
                }

                if(valid)
                {
                    inserted = true;
                    m_positions.push_back(sample);
                    active_list.push_back(m_positions.size() - 1);
                    grid[detail::cell2index(sample_index, num_cells)] = m_positions.size() - 1;
                    rotation                                          = detail::sample_rotation(gen, distrib);
                    m_rotations.push_back(rotation);

                    bool hitx, hity, hitz = false;
                    glm::vec3 boundary_sample;
                    hitx = detail::check_boundary(sample.x, m_voxel_length, radius / 2.0f, boundary_sample.x);
                    hity = detail::check_boundary(sample.y, m_voxel_length, radius / 2.0f, boundary_sample.y);
                    hitz = detail::check_boundary(sample.z, m_voxel_length, radius / 2.0f, boundary_sample.z);

                    for(uint32_t p = 0; p <= hitx; ++p)
                    {
                        for(uint32_t q = 0; q <= hity; ++q)
                        {
                            for(uint32_t l = 0; l <= hitz; ++l)
                            {
                                // Don't add default sample
                                if(p == 0 && q == 0 && l == 0) continue;
                                glm::vec3 new_sample = glm::vec3(p == 1 ? boundary_sample.x : sample.x,
                                                                 q == 1 ? boundary_sample.y : sample.y,
                                                                 l == 1 ? boundary_sample.z : sample.z);
                                boundary_spheres.push_back(new_sample);
                                boundary_rotations.push_back(rotation);
                            }
                        }
                    }

                    break;
                }
            }

            if(!inserted)
            {
                active_list.erase(std::next(active_list.begin(), active_index));
            }
        }
        uint32_t unique_spheres = m_positions.size();
        m_positions.insert(m_positions.end(), boundary_spheres.begin(), boundary_spheres.end());
        m_rotations.insert(m_rotations.end(), boundary_rotations.begin(), boundary_rotations.end());
        m_num_grains = m_positions.size();

        std::cout << "Sampled " << m_num_grains << " points with " << boundary_spheres.size()
                  << " being extra boundary grains" << std::endl;
        float packing_rate = static_cast<float>(unique_spheres) *
                             (4.0f / 3.0f * glm::pi<float>() * m_grain_size * m_grain_size * m_grain_size) /
                             (m_voxel_length * m_voxel_length * m_voxel_length);
        std::cout << "Packing rate: " << packing_rate << std::endl;

        delete[] grid;

        for(auto it = m_positions.begin(); it != m_positions.end();)
        {
            if((*it).x < m_grain_size - m_voxel_length / 2.0f)
            {
                it = m_positions.erase(it);
            }
            else if((*it).y < m_grain_size - m_voxel_length / 2.0f)
            {
                it = m_positions.erase(it);
            }
            else if((*it).z < m_grain_size - m_voxel_length / 2.0f)
            {
                it = m_positions.erase(it);
            }
            else
                ++it;
        }
    }

    struct Selector
    {
        using value_type = float;
        Selector()       = default;
        value_type select(const float& x) const { return x; }
    };

    std::vector<atcg::Instance> updateInstances()
    {
        std::vector<atcg::Instance> instances;
        for(int j = 0; j < height; ++j)
        {
            for(int i = 0; i < num_in_circle; ++i)
            {
                float angle   = 2.0f * glm::pi<float>() * i / num_in_circle;
                glm::vec3 pos = glm::vec3(radius * glm::cos(-angle), j * distance, radius * glm::sin(-angle));

                atcg::Instance instance;
                instance.color = glm::vec3(0, 15.0f / 255.0f, 0.0f);
                instance.model = glm::translate(pos) * glm::rotate(angle, glm::vec3(0, 1, 0)) *
                                 glm::rotate(elevation, glm::vec3(0, 0, 1));

                instances.push_back(instance);
            }
        }

        return instances;
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Application::get()->enableDockSpace(true);
        atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));
        atcg::Renderer::toggleCulling(false);
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = atcg::make_ref<atcg::FirstPersonController>(aspect_ratio);

        atcg::ref_ptr<atcg::Graph> grain    = atcg::IO::read_mesh("../Meshes/spruce_needle.obj");
        atcg::ref_ptr<atcg::Graph> sphere   = atcg::IO::read_mesh("res/sphere_low.obj");
        atcg::ref_ptr<atcg::Graph> cylinder = atcg::IO::read_mesh("res/cylinder.obj");

        // atcg::ref_ptr<atcg::Graph> bowl      = atcg::IO::read_mesh("../Meshes/bowl.obj");
        // atcg::ref_ptr<atcg::Graph> aggregate = atcg::IO::read_mesh("../Meshes/aggregate_rice_real.obj");

        // std::ifstream transform_file("../CVAE/data/Transforms_salt_spheres.bin", std::ios::in | std::ios::binary);
        // std::vector<char> transform_buffer(std::istreambuf_iterator<char>(transform_file), {});
        // float* transforms = reinterpret_cast<float*>(transform_buffer.data());

        // size_t num_instances =
        //     transform_buffer.size() / (sizeof(float) * 4 * 4);    // 16 floats for a transformation matrix

        // std::vector<atcg::Instance> instances;

        // for(int i = 0; i < num_instances; ++i)
        // {
        //     float* current_transform = transforms + 4 * 4 * i;
        //     glm::mat4 transform      = glm::transpose(glm::make_mat4(current_transform));

        //     // transform[0] *= 10.0f;
        //     // transform[1] *= 10.0f;
        //     // transform[2] *= 10.0f;

        //     atcg::Instance instance = {transform, glm::vec3(1.2903, 0.558, 0.1328)};

        //     instances.push_back(instance);
        // }

        scene = atcg::make_ref<atcg::Scene>();

        // std::ifstream input1("C:/Users/zingsheim/Downloads/packing.xyzd", std::ios::binary | std::ios::in);
        // std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input1), {});
        // glm::dvec4* data = reinterpret_cast<glm::dvec4*>(buffer.data());

        // std::cout << buffer.size() / (sizeof(double) * 4) << "\n";
        // for(uint32_t i = 0; i < buffer.size() / (sizeof(double) * 4); ++i)
        // {
        //     glm::vec3 current_point = data[i];
        //     current_point = glm::scale(glm::vec3(2.0f)) * glm::translate(glm::vec3(-20.0823593086113f / 2.0f)) *
        //                     glm::vec4(current_point, 1.0f);
        //     m_positions.push_back(current_point);
        // }

        // // Create acc structure for bottle
        // atcg::ref_ptr<atcg::TriMesh> bottle_mesh = atcg::make_ref<atcg::TriMesh>();
        // OpenMesh::IO::read_mesh(*bottle_mesh.get(), "../Meshes/salt_inner_300.obj");

        // for(auto vit = bottle_mesh->vertices_begin(); vit != bottle_mesh->vertices_end(); ++vit)
        // {
        //     bottle_mesh->set_point(*vit, glm::scale(glm::vec3(5.0f)) * glm::vec4(bottle_mesh->point(*vit), 1.0f));
        // }

        // atcg::Entity bottle_entity = scene->createEntity("BottleInner");
        // {
        //     auto& transform = bottle_entity.addComponent<atcg::TransformComponent>();
        //     // transform.setRotation(glm::vec3(0.0f, -glm::pi<float>() / 2.0f, 0.0f));
        //     // transform.setScale(glm::vec3(5.0f));
        //     bottle_entity.addComponent<atcg::GeometryComponent>(atcg::Graph::createTriangleMesh(bottle_mesh));
        //     auto& renderer   = bottle_entity.addComponent<atcg::MeshRenderComponent>();
        //     renderer.visible = false;
        //     atcg::Tracing::prepareAccelerationStructure(bottle_entity);
        // }

        // {
        //     atcg::Entity entity = scene->createEntity("Bottle");
        //     auto& transform     = entity.addComponent<atcg::TransformComponent>();
        //     // transform.setRotation(glm::vec3(0.0f, -glm::pi<float>() / 2.0f, 0.0f));
        //     // transform.setScale(glm::vec3(5.0f));
        //     entity.addComponent<atcg::GeometryComponent>(atcg::IO::read_mesh("../Meshes/salt_300.obj"));
        //     auto& renderer = entity.addComponent<atcg::MeshRenderComponent>();
        // }


        // std::ifstream input2("../Meshes/bottle_inner_sdf.bin", std::ios::binary);

        // // copies all data into buffer
        // std::vector<unsigned char> buffer2(std::istreambuf_iterator<char>(input2), {});

        // atcg::Grid<float> grid(glm::vec3(0), 128, 2.0f / 127.0f);
        // grid.setData(reinterpret_cast<float*>(buffer2.data()));

        // // poisson_sample();
        // std::vector<atcg::Vertex> vertices_cloud;
        // std::vector<glm::mat4> models;

        // for(int k = -6; k < 7; ++k)
        // {
        //     for(int j = -15; j < 16; ++j)
        //     {
        //         for(int l = -6; l < 7; ++l)
        //         {
        //             for(uint32_t i = 0; i < m_positions.size(); ++i)
        //             {
        //                 glm::vec3 grain_pos = m_positions[i] + glm::vec3(k, j, l) * m_voxel_length;
        //                 // glm::vec3 p         = glm::scale(glm::vec3(1.0f / 500.0f)) *
        //                 //               glm::rotate(-glm::pi<float>() / 2.0f, glm::vec3(0, 1, 0)) *
        //                 //               glm::vec4(grain_pos, 1.0f);
        //                 // if(!grid.insideVolume(p)) continue;
        //                 // float sdf = grid.readInterpolated<Selector>(p, Selector());
        //                 // if(sdf + 2.0f / 500.0f <= 0.0f)
        //                 // {
        //                 //     vertices_cloud.push_back(atcg::Vertex {grain_pos, glm::vec3(1), glm::vec3(1)});
        //                 //     models.push_back(glm::transpose(glm::translate(grain_pos)));
        //                 // }
        //                 glm::vec3 dir = glm::normalize(glm::vec3(1.0f));
        //                 atcg::Tracing::HitInfo isect =
        //                     atcg::Tracing::traceRay(bottle_entity, grain_pos, dir, 0.0f, 1e6f);

        //                 if(isect.hit)
        //                 {
        //                     glm::vec3 triangle_normal =
        //                         bottle_mesh->calc_face_normal(atcg::TriMesh::FaceHandle(isect.primitive_idx));
        //                     if(glm::dot(triangle_normal, dir) > 0.0f)
        //                     {
        //                         vertices_cloud.push_back(atcg::Vertex {grain_pos, glm::vec3(1), glm::vec3(1)});
        //                         glm::vec3 rotation_axis = glm::sphericalRand(1.0f);
        //                         float angle             = glm::linearRand(0.0f, 2.0f * glm::pi<float>());
        //                         glm::mat4 rotation      = glm::rotate(angle, rotation_axis);
        //                         models.push_back(glm::transpose(glm::translate(grain_pos) * rotation));
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }

        // std::vector<atcg::Vertex> vertices;
        // for(uint32_t i = 0; i < grid.voxels_per_volume(); ++i)
        // {
        //     glm::vec3 p = grid.voxel2position(grid.index2voxel(i));
        //     float sdf   = grid.readInterpolated<Selector>(p, Selector());
        //     p           = glm::rotate(glm::pi<float>() / 2.0f, glm::vec3(0, 1, 0)) * glm::scale(glm::vec3(100)) *
        //         glm::vec4(p, 1.0f);
        //     if(sdf <= 0.0f) { vertices.push_back(atcg::Vertex {p, glm::vec3(1), glm::vec3(1)}); }
        // }

        // std::cout << (vertices_cloud.size()) << "\n";

        // atcg::ref_ptr<atcg::Graph> pc = atcg::Graph::createPointCloud(vertices_cloud);
        // {
        //     atcg::Entity entity = scene->createEntity("Grains");
        //     auto& transform     = entity.addComponent<atcg::TransformComponent>();
        //     entity.addComponent<atcg::GeometryComponent>(pc);
        //     auto& renderer      = entity.addComponent<atcg::PointRenderComponent>();
        //     renderer.point_size = 1.0f;
        //     // renderer.shader     = atcg::ShaderManager::getShader("flat");
        //     renderer.color = glm::vec3(1, 0, 0);
        // }

        // {
        //     atcg::Entity entity = scene->createEntity("Offset");
        //     auto& transform     = entity.addComponent<atcg::TransformComponent>();
        //     transform.setPosition(glm::vec3(10, 0, 0));
        //     entity.addComponent<atcg::GeometryComponent>(pc);
        //     auto& renderer      = entity.addComponent<atcg::PointSphereRenderComponent>();
        //     renderer.point_size = 1.0f;
        //     // renderer.shader     = atcg::ShaderManager::getShader("flat");
        //     hovered_entity = entity;
        // }

        // {
        //     atcg::Entity entity = scene->createEntity();
        //     auto& transform     = entity.addComponent<atcg::TransformComponent>();
        //     // transform.setPosition(glm::vec3(0, 50, 0));
        //     entity.addComponent<atcg::GeometryComponent>(grain);
        //     auto& renderer = entity.addComponent<atcg::MeshRenderComponent>();
        //     renderer.color = glm::vec3(0, 15.0f / 255.0f, 0.0f);
        // }

        std::vector<atcg::Instance> instances = updateInstances();


        {
            leaf_entity     = scene->createEntity("Leafs");
            auto& transform = leaf_entity.addComponent<atcg::TransformComponent>();
            // transform.setPosition(glm::vec3(0, 50, 0));
            leaf_entity.addComponent<atcg::GeometryComponent>(grain);
            leaf_entity.addComponent<atcg::InstanceRenderComponent>(instances);
        }

        {
            cylinder_entity = scene->createEntity("Branch");
            auto& transform = cylinder_entity.addComponent<atcg::TransformComponent>();
            // transform.setPosition(glm::vec3(0, 50, 0));
            cylinder_entity.addComponent<atcg::GeometryComponent>(cylinder);
            auto& renderer = cylinder_entity.addComponent<atcg::MeshRenderComponent>();
            renderer.material.setDiffuseColor(glm::vec4(19.0f / 255.0f, 9.0f / 255.0f, 0.0f, 1.0f));

            float branch_radius = radius - glm::cos(elevation) - 0.01f;
            float branch_height = height * distance;
            transform.setScale(glm::vec3(branch_radius, branch_height / 2.0f, branch_radius));
            transform.setPosition(glm::vec3(0, branch_height / 2.0f, 0));
        }

        // {
        //     atcg::Entity entity = scene->createEntity();
        //     auto& transform     = entity.addComponent<atcg::TransformComponent>();
        //     transform.setPosition(glm::vec3(0, 20, 0));
        //     transform.setScale(glm::vec3(0.4));
        //     entity.addComponent<atcg::GeometryComponent>(bowl);
        //     entity.addComponent<atcg::MeshRenderComponent>();
        // }

        // {
        //     atcg::Entity entity = scene->createEntity();
        //     auto& transform     = entity.addComponent<atcg::TransformComponent>();
        //     transform.setPosition(glm::vec3(0, 50, 0));
        //     entity.addComponent<atcg::GeometryComponent>(aggregate);
        //     entity.addComponent<atcg::MeshRenderComponent>();
        // }

        // std::ofstream summary_file("Transforms_bottle.bin", std::ios::out | std::ios::binary);
        // summary_file.write((char*)models.data(), sizeof(glm::mat4) * models.size());
        // summary_file.close();

        panel = atcg::SceneHierarchyPanel(scene);
        panel.selectEntity(hovered_entity);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        atcg::Renderer::drawCADGrid(camera_controller->getCamera());

        atcg::Renderer::draw(scene, camera_controller->getCamera());

        dt = delta_time;
    }

    virtual void onImGuiRender() override
    {
        ImGui::BeginMainMenuBar();

        if(ImGui::BeginMenu("Rendering"))
        {
            ImGui::MenuItem("Show Render Settings", nullptr, &show_render_settings);
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();

        if(show_render_settings)
        {
            ImGui::Begin("Settings", &show_render_settings);

            std::stringstream ss;
            ss << "FPS: " << 1.0f / dt << " | " << dt << " ms\n";
            ImGui::Text(ss.str().c_str());

            ImGui::End();
        }

        panel.renderPanel();
        hovered_entity = panel.getSelectedEntity();

        // Gizmo test

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2 {0, 0});
        const auto& window       = atcg::Application::get()->getWindow();
        glm::ivec2 window_pos    = window->getPosition();
        glm::ivec2 viewport_pos  = atcg::Application::get()->getViewportPosition();
        glm::ivec2 viewport_size = atcg::Application::get()->getViewportSize();
        ImGui::Begin("Viewport");
        if(hovered_entity)
        {
            ImGuizmo::SetOrthographic(false);
            ImGuizmo::BeginFrame();
            ImGuizmo::SetDrawlist();

            ImGuizmo::SetRect(window_pos.x + viewport_pos.x,
                              window_pos.y + viewport_pos.y,
                              viewport_size.x,
                              viewport_size.y);

            glm::mat4 camera_projection = camera_controller->getCamera()->getProjection();
            glm::mat4 camera_view       = camera_controller->getCamera()->getView();

            atcg::TransformComponent& transform = hovered_entity.getComponent<atcg::TransformComponent>();
            glm::mat4 model                     = transform.getModel();
            ImGuizmo::Manipulate(glm::value_ptr(camera_view),
                                 glm::value_ptr(camera_projection),
                                 current_operation,
                                 ImGuizmo::LOCAL,
                                 glm::value_ptr(model));
            transform.setModel(model);
        }
        ImGui::End();
        ImGui::PopStyleVar();

        ImGui::Begin("Needle Settings");

        bool update = ImGui::InputInt("Height", &height);
        update |= ImGui::InputInt("Num Leaves", &num_in_circle);
        update |= ImGui::InputFloat("Leaf distance", &distance);
        update |= ImGui::InputFloat("Radius", &radius);
        update |= ImGui::SliderFloat("Rotation", &elevation, 0.0f, glm::pi<float>() / 2.0f);

        if(update)
        {
            std::vector<atcg::Instance> instances = updateInstances();

            leaf_entity.removeComponent<atcg::InstanceRenderComponent>();
            leaf_entity.addComponent<atcg::InstanceRenderComponent>(instances);
            auto& geometry = leaf_entity.getComponent<atcg::GeometryComponent>();
            geometry.graph->getVerticesArray()->popVertexBuffer();

            auto& transform     = cylinder_entity.getComponent<atcg::TransformComponent>();
            float branch_radius = radius - glm::cos(elevation) - 0.01f;
            float branch_height = height * distance;
            transform.setScale(glm::vec3(branch_radius, branch_height / 2.0f, branch_radius));
            transform.setPosition(glm::vec3(0, branch_height / 2.0f, 0));
        }

        if(ImGui::Button("Save"))
        {
            std::vector<atcg::Instance> instances = updateInstances();

            std::vector<glm::mat4> models;
            for(int i = 0; i < instances.size(); ++i)
            {
                models.push_back(glm::transpose(instances[i].model));
            }
            std::ofstream summary_file("Transforms_needle.bin", std::ios::out | std::ios::binary);
            summary_file.write((char*)models.data(), sizeof(glm::mat4) * models.size());
            summary_file.close();
        }

        ImGui::End();
    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
        dispatcher.dispatch<atcg::KeyPressedEvent>(ATCG_BIND_EVENT_FN(GrainLayer::onKeyPressed));
        dispatcher.dispatch<atcg::ViewportResizeEvent>(ATCG_BIND_EVENT_FN(GrainLayer::onViewportResized));
        dispatcher.dispatch<atcg::MouseMovedEvent>(ATCG_BIND_EVENT_FN(GrainLayer::onMouseMoved));
        dispatcher.dispatch<atcg::MouseButtonPressedEvent>(ATCG_BIND_EVENT_FN(GrainLayer::onMousePressed));
    }

    bool onMousePressed(atcg::MouseButtonPressedEvent* event)
    {
        if(in_viewport && event->getMouseButton() == GLFW_MOUSE_BUTTON_LEFT && !ImGuizmo::IsOver())
        {
            int id         = atcg::Renderer::getEntityIndex(mouse_pos);
            hovered_entity = id == -1 ? atcg::Entity() : atcg::Entity((entt::entity)id, scene.get());
            panel.selectEntity(hovered_entity);
        }
        return true;
    }

    bool onMouseMoved(atcg::MouseMovedEvent* event)
    {
        const atcg::Application* app = atcg::Application::get();
        glm::ivec2 offset            = app->getViewportPosition();
        int height                   = app->getViewportSize().y;
        mouse_pos                    = glm::vec2(event->getX() - offset.x, height - (event->getY() - offset.y));

        in_viewport =
            mouse_pos.x >= 0 && mouse_pos.y >= 0 && mouse_pos.y < height && mouse_pos.x < app->getViewportSize().x;

        return false;
    }

    bool onKeyPressed(atcg::KeyPressedEvent* event)
    {
        if(event->getKeyCode() == GLFW_KEY_T)
        {
            current_operation = ImGuizmo::OPERATION::TRANSLATE;
        }
        if(event->getKeyCode() == GLFW_KEY_R)
        {
            current_operation = ImGuizmo::OPERATION::ROTATE;
        }
        if(event->getKeyCode() == GLFW_KEY_S)
        {
            current_operation = ImGuizmo::OPERATION::SCALE;
        }
        // if(event->getKeyCode() == GLFW_KEY_L) { camera_controller->getCamera()->setLookAt(sphere->getPosition());
        // }

        return true;
    }

    bool onViewportResized(atcg::ViewportResizeEvent* event)
    {
        atcg::WindowResizeEvent resize_event(event->getWidth(), event->getHeight());
        camera_controller->onEvent(&resize_event);
        return false;
    }

private:
    atcg::ref_ptr<atcg::Scene> scene;
    atcg::ref_ptr<atcg::FirstPersonController> camera_controller;
    atcg::Entity hovered_entity;
    atcg::SceneHierarchyPanel<atcg::ComponentGUIHandler> panel;

    std::vector<glm::vec3> m_positions;
    std::vector<glm::vec3> m_rotations;
    float m_grain_size   = 1.0f;
    float m_voxel_length = 2.0f * 20.0823593086113f;
    int m_num_grains     = 0;

    atcg::Entity leaf_entity;
    atcg::Entity cylinder_entity;
    int height        = 40;
    int num_in_circle = 8;
    float distance    = 0.35f;
    float radius      = 0.8f;
    float elevation   = glm::pi<float>() / 4.0f;

    bool show_render_settings = true;

    glm::vec2 mouse_pos;
    bool in_viewport = false;

    float dt = 1.0f / 60.0f;

    ImGuizmo::OPERATION current_operation = ImGuizmo::OPERATION::TRANSLATE;
};

class Grains : public atcg::Application
{
public:
    Grains() : atcg::Application() { pushLayer(new GrainLayer("Layer")); }

    ~Grains() {}
};

atcg::Application* atcg::createApplication()
{
    return new Grains;
}