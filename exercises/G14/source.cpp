#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <queue>

#include <numeric>
#include <random>

#include <glm/gtc/matrix_transform.hpp>

class G14Layer : public atcg::Layer
{
public:
    G14Layer(const std::string& name) : atcg::Layer(name) {}

    std::vector<uint32_t> find_correspondences(const std::shared_ptr<atcg::Mesh>& source,
                                               const std::shared_ptr<atcg::Mesh>& target)
    {
        const uint32_t n = source->n_vertices();
        std::vector<uint32_t> correspondences(n);

        for(uint32_t i = 0; i < n; ++i)
        {
            atcg::Mesh::Point q = source->point(atcg::Mesh::VertexHandle(i));
            int minj            = 0;
            double min_distance = std::numeric_limits<double>::infinity();
            for(uint32_t j = 0; j < n; ++j)
            {
                atcg::Mesh::Point p = target->point(atcg::Mesh::VertexHandle(j));
                double distance     = (p - q).norm();
                if(distance < min_distance)
                {
                    min_distance = distance;
                    minj         = j;
                }
            }

            correspondences[i] = minj;
        }

        return correspondences;
    }

    std::pair<atcg::Mesh::Point, atcg::Mesh::Point> calculate_center_of_mass(const std::shared_ptr<atcg::Mesh>& source,
                                                                             const std::shared_ptr<atcg::Mesh>& target,
                                                                             const std::vector<uint32_t>& c)
    {
        const uint32_t n = source->n_vertices();
        atcg::Mesh::Point q_bar {0, 0, 0};
        atcg::Mesh::Point p_bar {0, 0, 0};

        for(uint32_t i = 0; i < n; ++i)
        {
            q_bar += source->point(atcg::Mesh::VertexHandle(i));
            p_bar += target->point(atcg::Mesh::VertexHandle(c[i]));
        }

        return std::make_pair(q_bar / n, p_bar / n);
    }

    void update_rotation(const std::shared_ptr<atcg::Mesh>& source,
                         const std::shared_ptr<atcg::Mesh>& target,
                         const atcg::Mesh::Point& q_bar,
                         const atcg::Mesh::Point& p_bar,
                         const std::vector<uint32_t>& c)
    {
        const uint32_t n = source->n_vertices();

        Eigen::Matrix3d H;
        H.setZero();

        for(uint32_t i = 0; i < n; ++i)
        {
            atcg::Mesh::Point q_temp = source->point(atcg::Mesh::VertexHandle(i)) - q_bar;
            Eigen::Vector3d q_(q_temp[0], q_temp[1], q_temp[2]);

            atcg::Mesh::Point p_temp = target->point(atcg::Mesh::VertexHandle(c[i])) - p_bar;
            Eigen::Vector3d p_(p_temp[0], p_temp[1], p_temp[2]);

            H += q_ * p_.transpose();
        }

        Eigen::JacobiSVD<Eigen::Matrix3d, Eigen::ComputeFullU | Eigen::ComputeFullV> solver(H);
        R = solver.matrixV() * solver.matrixU().transpose();
    }

    void update_translation(const atcg::Mesh::Point& q_bar, const atcg::Mesh::Point& p_bar)
    {
        Eigen::Vector3d q_temp(q_bar[0], q_bar[1], q_bar[2]);
        Eigen::Vector3d p_temp(p_bar[0], p_bar[1], p_bar[2]);

        t = p_temp - R * q_temp;
    }

    void apply_transform(const std::shared_ptr<atcg::Mesh>& source)
    {
        for(auto v_it = source->vertices_begin(); v_it != source->vertices_end(); ++v_it)
        {
            atcg::Mesh::Point q = source->point(*v_it);
            Eigen::Vector3d q_temp(q[0], q[1], q[2]);

            q_temp = R * q_temp + t;

            source->set_point(*v_it, atcg::Mesh::Point {q_temp(0), q_temp(1), q_temp(2)});
        }
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = std::make_shared<atcg::CameraController>(aspect_ratio);

        R.setIdentity();
        t.setZero();

        mesh_source = atcg::IO::read_mesh("res/suzanne_blender.obj");
        mesh_target = atcg::IO::read_mesh("res/suzanne_blender.obj");

        mesh_source->request_vertex_colors();
        mesh_target->request_vertex_colors();

        for(uint32_t i = 0; i < mesh_target->n_vertices(); ++i)
        {
            mesh_source->set_color(atcg::Mesh::VertexHandle(i), atcg::Mesh::Color {255, 0, 0});
            mesh_target->set_color(atcg::Mesh::VertexHandle(i), atcg::Mesh::Color {0, 255, 0});

            atcg::Mesh::Point p_ = mesh_source->point(atcg::Mesh::VertexHandle(i));
            glm::vec4 p(p_[0], p_[1], p_[2], 1.0);

            glm::mat4 T = glm::translate(
                glm::rotate(glm::mat4(1), glm::pi<float>() / 8.0f, glm::normalize(glm::vec3(0.0f, -0.1f, 0.8f))),
                glm::vec3(1.0f, -0.2f, 0.5f));
            p = T * p;

            mesh_source->set_point(atcg::Mesh::VertexHandle(i), {p.x, p.y, p.z});
        }

        mesh_source->uploadData();
        mesh_target->uploadData();
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        if(mesh_target && render_faces_target)
            atcg::Renderer::draw(mesh_target, atcg::ShaderManager::getShader("base"), camera_controller->getCamera());

        if(mesh_target && render_points_target)
            atcg::Renderer::drawPoints(mesh_target,
                                       glm::vec3(0),
                                       atcg::ShaderManager::getShader("base"),
                                       camera_controller->getCamera());

        if(mesh_target && render_edges_target)
            atcg::Renderer::drawLines(mesh_target, glm::vec3(1), camera_controller->getCamera());

        if(mesh_source && render_faces_source)
            atcg::Renderer::draw(mesh_source, atcg::ShaderManager::getShader("base"), camera_controller->getCamera());

        if(mesh_source && render_points_source)
            atcg::Renderer::drawPoints(mesh_source,
                                       glm::vec3(0),
                                       atcg::ShaderManager::getShader("base"),
                                       camera_controller->getCamera());

        if(mesh_source && render_edges_source)
            atcg::Renderer::drawLines(mesh_source, glm::vec3(1), camera_controller->getCamera());
    }

    virtual void onImGuiRender() override
    {
        ImGui::BeginMainMenuBar();

        if(ImGui::BeginMenu("Rendering"))
        {
            ImGui::MenuItem("Show Render Settings", nullptr, &show_render_settings);
            ImGui::MenuItem("Show Registration Settings", nullptr, &show_registration_settings);

            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();

        if(show_render_settings)
        {
            ImGui::Begin("Settings", &show_render_settings);

            ImGui::Checkbox("Render Target Vertices", &render_points_target);
            ImGui::Checkbox("Render Target Edges", &render_edges_target);
            ImGui::Checkbox("Render Target Mesh", &render_faces_target);

            ImGui::Checkbox("Render Source Vertices", &render_points_source);
            ImGui::Checkbox("Render Source Edges", &render_edges_source);
            ImGui::Checkbox("Render Source Mesh", &render_faces_source);
            ImGui::End();
        }

        if(show_registration_settings)
        {
            ImGui::Begin("Registration Settings", &show_registration_settings);

            if(ImGui::Button("Step"))
            {
                std::vector<uint32_t> correspondences = find_correspondences(mesh_source, mesh_target);
                auto [q_bar, p_bar] = calculate_center_of_mass(mesh_source, mesh_target, correspondences);
                update_rotation(mesh_source, mesh_target, q_bar, p_bar, correspondences);
                update_translation(q_bar, p_bar);
                apply_transform(mesh_source);
                mesh_source->uploadData();
            }

            ImGui::End();
        }
    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override { camera_controller->onEvent(event); }

private:
    std::shared_ptr<atcg::CameraController> camera_controller;
    std::shared_ptr<atcg::Mesh> mesh_target;
    std::shared_ptr<atcg::Mesh> mesh_source;

    bool show_render_settings = true;
    bool render_faces_target  = true;
    bool render_points_target = false;
    bool render_edges_target  = false;

    bool render_faces_source  = true;
    bool render_points_source = false;
    bool render_edges_source  = false;

    bool show_registration_settings = true;

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
};

class G14 : public atcg::Application
{
public:
    G14() : atcg::Application() { pushLayer(new G14Layer("Layer")); }

    ~G14() {}
};

atcg::Application* atcg::createApplication()
{
    return new G14;
}