#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>

class G03Layer : public atcg::Layer
{
public:
    G03Layer(const std::string& name) : atcg::Layer(name) {}

    glm::vec3
    from_barycentric_cooridnates(const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3, const float* barys)
    {
        return p1 * barys[0] + p2 * barys[1] + p3 * barys[2];
    }

    glm::vec3 circumcenter(const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3)
    {
        float a = glm::length(p2 - p3);
        float b = glm::length(p3 - p1);
        float c = glm::length(p1 - p2);
        float bary[3];
        bary[0]   = a * a * (b * b + c * c - a * a);
        bary[1]   = b * b * (c * c + a * a - b * b);
        bary[2]   = c * c * (a * a + b * b - c * c);
        float sum = bary[0] + bary[1] + bary[2];
        bary[0] /= sum;
        bary[1] /= sum;
        bary[2] /= sum;
        return from_barycentric_cooridnates(p1, p2, p3, bary);
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        // This data is used for rendering
        // The vbo holds the actual points to render
        // The vao holds information about the buffer structure. Here we only have a float3 for the position
        vao = std::make_shared<atcg::VertexArray>();

        vbo = std::make_shared<atcg::VertexBuffer>(static_cast<uint32_t>(max_num_points * sizeof(float) * 3));
        vbo->setLayout({{atcg::ShaderDataType::Float3, "aPosition"}});
        vao->addVertexBuffer(vbo);

        ibo = std::make_shared<atcg::IndexBuffer>(max_num_points);
        vao->setIndexBuffer(ibo);

        vao_bary = std::make_shared<atcg::VertexArray>();

        vbo_bary = std::make_shared<atcg::VertexBuffer>(static_cast<uint32_t>(sizeof(float) * 3));
        vbo_bary->setLayout({{atcg::ShaderDataType::Float3, "aPosition"}});
        vao_bary->addVertexBuffer(vbo_bary);

        uint32_t z = 0;
        ibo_bary   = std::make_shared<atcg::IndexBuffer>(&z, 1);
        vao_bary->setIndexBuffer(ibo_bary);

        vao_cc = std::make_shared<atcg::VertexArray>();

        vbo_cc = std::make_shared<atcg::VertexBuffer>(static_cast<uint32_t>(sizeof(float) * 3));
        vbo_cc->setLayout({{atcg::ShaderDataType::Float3, "aPosition"}});
        vao_cc->addVertexBuffer(vbo_cc);

        ibo_cc = std::make_shared<atcg::IndexBuffer>(&z, 1);
        vao_cc->setIndexBuffer(ibo_cc);

        barys = new float[3] {0, 0, 0};

        const auto& window = atcg::Application::get()->getWindow();
        camera             = std::make_shared<atcg::OrthographicCamera>(0.f,
                                                            static_cast<float>(window->getWidth()),
                                                            0.f,
                                                            static_cast<float>(window->getHeight()));
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        atcg::Renderer::clear();

        if(points.size() > 0)
            atcg::Renderer::drawPoints(vao, glm::vec3(0), atcg::ShaderManager::getShader("flat"), camera);

        if(points.size() > 0)
        {
            atcg::ShaderManager::getShader("flat")->setVec3("flat_color", glm::vec3(0.8f));
            atcg::Renderer::draw(vao, atcg::ShaderManager::getShader("flat"), camera);
        }

        if(bary_set)
            atcg::Renderer::drawPoints(vao_bary, glm::vec3(1, 0, 0), atcg::ShaderManager::getShader("flat"), camera);

        if(cc_set)
        {
            atcg::Renderer::drawPoints(vao_cc, glm::vec3(0, 1, 0), atcg::ShaderManager::getShader("flat"), camera);
            atcg::Renderer::drawCircle(circum_center,
                                       glm::length(circum_center - points[0]),
                                       glm::vec3(0, 1, 0),
                                       camera);
        }
    }

    virtual void onImGuiRender() override
    {
        ImGui::BeginMainMenuBar();

        if(ImGui::BeginMenu("Rendering"))
        {
            ImGui::MenuItem("Show Render Settings", nullptr, &show_render_settings);
            ImGui::EndMenu();
        }

        if(ImGui::BeginMenu("Exercise"))
        {
            ImGui::MenuItem("Show Center Settings", nullptr, &show_center_settings);
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();

        if(show_render_settings)
        {
            ImGui::Begin("Settings", &show_render_settings);

            if(ImGui::Button("Clear all"))
            {
                points.clear();
                indices.clear();
                bary_set = false;
                cc_set   = false;
            }

            ImGui::End();
        }

        if(show_center_settings)
        {
            ImGui::Begin("Settings Center", &show_center_settings);

            if(ImGui::SliderFloat3("Barycentric coordinates", barys, 0.0f, 1.0f))
            {
                if(points.size() == 3)
                {
                    glm::vec3 p = from_barycentric_cooridnates(points[0], points[1], points[2], barys);
                    p.z += 0.01f;
                    vbo_bary->setData(reinterpret_cast<float*>(&p), 3 * sizeof(float));
                    bary_set = true;
                }
            }

            if(ImGui::Button("Calculate circumcenter"))
            {
                glm::vec3 p = circumcenter(points[0], points[1], points[2]);
                p.z += 0.01f;
                vbo_cc->setData(reinterpret_cast<float*>(&p), 3 * sizeof(float));
                circum_center = p;
                cc_set        = true;
            }

            ImGui::End();
        }
    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override
    {
        atcg::EventDispatcher distpatcher(event);
        if(atcg::Input::isKeyPressed(GLFW_KEY_LEFT_SHIFT))
        {
            distpatcher.dispatch<atcg::MouseButtonPressedEvent>(ATCG_BIND_EVENT_FN(G03Layer::onMousePressed));
        }
        distpatcher.dispatch<atcg::WindowResizeEvent>(ATCG_BIND_EVENT_FN(G03Layer::onWindowResized));
    }

    bool onMousePressed(atcg::MouseButtonPressedEvent* e)
    {
        if(points.size() >= 3)
        {
            printf("Max numbers of points reached\n");
            return false;
        }

        const auto& window = atcg::Application::get()->getWindow();

        float width         = (float)window->getWidth();
        float height        = (float)window->getHeight();
        glm::vec2 mouse_pos = atcg::Input::getMousePosition();

        float x_ndc = (static_cast<float>(mouse_pos.x)) / (static_cast<float>(width) / 2.0f) - 1.0f;
        float y_ndc =
            (static_cast<float>(height) - static_cast<float>(mouse_pos.y)) / (static_cast<float>(height) / 2.0f) - 1.0f;

        glm::vec4 world_pos(x_ndc, y_ndc, 0.0f, 1.0f);

        world_pos = glm::inverse(camera->getViewProjection()) * world_pos;
        world_pos /= world_pos.w;

        points.push_back(world_pos);

        // Update the rendering data
        vbo->setData(reinterpret_cast<float*>(points.data()), static_cast<uint32_t>(points.size() * 3 * sizeof(float)));
        size_t old_size = indices.size();
        indices.resize(points.size());
        std::iota(indices.begin() + old_size, indices.end(), static_cast<uint32_t>(old_size));

        ibo->setData(indices.data(), static_cast<uint32_t>(indices.size()));
        return true;
    }

    bool onWindowResized(atcg::WindowResizeEvent* event)
    {
        camera->setProjection(0, static_cast<float>(event->getWidth()), 0, static_cast<float>(event->getHeight()));
        return false;
    }

private:
    bool show_render_settings = false;

    bool show_center_settings = true;
    bool bary_set             = false;
    bool cc_set               = false;
    glm::vec3 circum_center;
    float* barys;

    std::shared_ptr<atcg::OrthographicCamera> camera;

    std::vector<glm::vec3> points;
    std::vector<uint32_t> indices;
    std::shared_ptr<atcg::VertexArray> vao;
    std::shared_ptr<atcg::VertexBuffer> vbo;
    std::shared_ptr<atcg::IndexBuffer> ibo;

    std::shared_ptr<atcg::VertexArray> vao_bary;
    std::shared_ptr<atcg::VertexBuffer> vbo_bary;
    std::shared_ptr<atcg::IndexBuffer> ibo_bary;

    std::shared_ptr<atcg::VertexArray> vao_cc;
    std::shared_ptr<atcg::VertexBuffer> vbo_cc;
    std::shared_ptr<atcg::IndexBuffer> ibo_cc;
    uint32_t max_num_points = 100;
};

class G03 : public atcg::Application
{
public:
    G03() : atcg::Application() { pushLayer(new G03Layer("Layer")); }

    ~G03() {}
};

atcg::Application* atcg::createApplication()
{
    return new G03;
}