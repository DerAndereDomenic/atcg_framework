#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <queue>

#include <numeric>

class G09Layer : public atcg::Layer
{
public:

    G09Layer(const std::string& name) : atcg::Layer(name) {}

    std::vector<float> linspace(float a, float b, uint32_t steps)
    {
        float step_size = (b-a) / (steps - 1);

        std::vector<float> space(steps);

        for(uint32_t i = 0; i < steps; ++i)
        {
            space[i] = (a + i * step_size);
        }

        return space;
    }

    std::shared_ptr<atcg::Mesh> triangulate(const std::vector<atcg::Mesh::Point>& points)
    {
        std::shared_ptr<atcg::Mesh> mesh = std::make_shared<atcg::Mesh>();
        
        std::vector<atcg::Mesh::VertexHandle> v_handles(points.size());

        for(uint32_t i = 0; i < points.size(); ++i)
            v_handles[i] = mesh->add_vertex({points[i][0], points[i][2], points[i][1]});

        uint32_t grid_size = static_cast<uint32_t>(std::sqrt(points.size())) - 1;

        for(uint32_t grid_x = 0; grid_x < grid_size; ++grid_x)
        {
            for(uint32_t grid_y = 0; grid_y < grid_size; ++grid_y)
            {
                auto v00 = v_handles[grid_x + (grid_size+1) * grid_y];
                auto v10 = v_handles[grid_x + 1 + (grid_size+1) * grid_y];
                auto v01 = v_handles[grid_x + (grid_size+1) * (grid_y + 1)];
                auto v11 = v_handles[grid_x + 1 + (grid_size+1) * (grid_y + 1)];

                mesh->add_face(v00, v01, v10);
                mesh->add_face(v10, v01, v11);
            }
        }

        return mesh;
    }

    struct BezierBase
    {
        float coefficients[4];
    };

    BezierBase getBezierCoefficients(float t)
    {
        BezierBase base;

        base.coefficients[0] = (1 - t) * (1 - t) * (1 - t);
        base.coefficients[1] = 3 * t * (1 - t) * (1 - t);
        base.coefficients[2] = 3 * t * t * (1 - t);
        base.coefficients[3] = t * t * t;

        return base;
    }

    std::vector<atcg::Mesh::Point> calculate_surface(const std::vector<atcg::Mesh::Point>& control_points, const std::vector<float>& samples)
    {
        std::vector<atcg::Mesh::Point> points;
        uint32_t m = 4, n = 4;

        for(float u : samples)
        {
            for(float v : samples)
            {
                BezierBase F = getBezierCoefficients(u);
                BezierBase G = getBezierCoefficients(v);

                atcg::Mesh::Point p{0,0,0};

                for(uint32_t i = 0; i < n; ++i)
                {
                    for(uint32_t j = 0; j < m; ++j)
                    {
                        p += control_points[j + m * i] * F.coefficients[i] * G.coefficients[j]; 
                    }
                }

                points.push_back(p);
            }
        }

        return points;
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        std::vector<float> X = linspace(-1,1,4);
        std::vector<float> Y = linspace(-1,1,4);
        std::vector<float> Z = {0.1f, 0.4f, -0.1f, 0.3f, 0.3f, 0.3f, 0.8f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.2f, 0.4f, 1.0f, 0.1f};
        frequencies = {0.10f,0.2f,0.3f,0.12f,0.1f,0.5f,0.5f,0.25f,0.13f,0.15f,0.2f,0.20f,0.9f,0.11f,0.8f,0.7f};

        for(uint32_t i = 0; i < Y.size(); ++i)
        {
            for(uint32_t j = 0; j < X.size(); ++j)
            {
                control_polygon.push_back({X[j], Y[i], Z[j + X.size() * i]});
            }
        }

        control_polygon_mesh = triangulate(control_polygon);
        control_polygon_mesh->uploadData();

        std::vector<float> sample_points = linspace(0,1, 100);

        std::vector<atcg::Mesh::Point> surface = calculate_surface(control_polygon, sample_points);
        mesh = triangulate(surface);
        mesh->uploadData();
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        t += delta_time;

        for(uint32_t i = 0; i < control_polygon.size(); ++i)
        {
            control_polygon[i][2] = std::sin(2.0f * static_cast<float>(M_PI) * frequencies[i] * t);
        }

        control_polygon_mesh = triangulate(control_polygon);
        std::vector<atcg::Mesh::Point> surface = calculate_surface(control_polygon, linspace(0,1,100));
        mesh = triangulate(surface);

        control_polygon_mesh->uploadData();
        mesh->uploadData();

        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        if(mesh && render_faces)
            atcg::Renderer::draw(mesh, atcg::ShaderManager::getShader("base"), camera_controller->getCamera());

        if(mesh && render_points)
            atcg::Renderer::drawPoints(mesh, glm::vec3(0), atcg::ShaderManager::getShader("base"), camera_controller->getCamera());

        if(mesh && render_edges)
            atcg::Renderer::drawLines(mesh, glm::vec3(1), camera_controller->getCamera());

        if(control_polygon_mesh && render_control_polygon)
            atcg::Renderer::drawLines(control_polygon_mesh, glm::vec3(1,0,0), camera_controller->getCamera());
    }

    virtual void onImGuiRender() override
    {
        ImGui::BeginMainMenuBar();

        if(ImGui::BeginMenu("Rendering"))
        {
            ImGui::MenuItem("Show Render Settings", nullptr, &show_render_settings);
            ImGui::MenuItem("Show Render Settings", nullptr, &show_surface_settings);

            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();

        if(show_render_settings)
        {
            ImGui::Begin("Settings", &show_render_settings);

            ImGui::Checkbox("Render Vertices", &render_points);
            ImGui::Checkbox("Render Edges", &render_edges);
            ImGui::Checkbox("Render Mesh", &render_faces);
            ImGui::Checkbox("Render Control Polygon", &render_control_polygon);
            ImGui::End();
        }

        if(show_surface_settings)
        {
            ImGui::Begin("Surface Settings", &show_surface_settings);
            for(uint32_t i = 0; i < frequencies.size(); ++i)
            {
                float& fr = frequencies[i];
                ImGui::SliderFloat(("Point " + std::to_string(i)).c_str(), &fr, 0, 1.0f);
            }
            ImGui::End();
        }

    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event& event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
    }

private:
    std::shared_ptr<atcg::CameraController> camera_controller;
    std::shared_ptr<atcg::Mesh> mesh;
    std::vector<atcg::Mesh::Point> control_polygon;
    std::vector<float> frequencies;
    std::shared_ptr<atcg::Mesh> control_polygon_mesh;

    bool show_render_settings = true;
    bool show_surface_settings = true;
    bool render_faces = true;
    bool render_points = false;
    bool render_edges = false;

    bool render_control_polygon = true;

    float t = 0.0f;
};

class G09 : public atcg::Application
{
    public:

    G09()
        :atcg::Application()
    {
        pushLayer(new G09Layer("Layer"));
    }

    ~G09() {}

};

atcg::Application* atcg::createApplication()
{
    return new G09;
}