#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <glfw/glfw3.h>
#include <imgui.h>

class G01Layer : public atcg::Layer
{
public:

    G01Layer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        vao = std::make_shared<atcg::VertexArray>();

        vbo = std::make_shared<atcg::VertexBuffer>(static_cast<uint32_t>(100 * sizeof(float) * 3));
        vbo->setLayout({{atcg::ShaderDataType::Float3, "aPosition"}});
        vao->addVertexBuffer(vbo);

        ibo = std::make_shared<atcg::IndexBuffer>(100);
        vao->setIndexBuffer(ibo);

        float aspect_ratio = (float)atcg::Application::get()->getWindow()->getWidth() / (float)atcg::Application::get()->getWindow()->getHeight();

        atcg::ShaderManager::addShaderFromName("bezier");
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        atcg::Renderer::clear();

        atcg::Renderer::drawPoints(vao, glm::vec3(0), atcg::ShaderManager::getShader("flat"));

        int discretization = 20;
        uint32_t point_index = 0;

        const auto& shader = atcg::ShaderManager::getShader("bezier");
        shader->use();
        vao->use();
        shader->setInt("discretization", discretization);
        shader->setVec3("flat_color", glm::vec3(1,0,0));
        if(points.size() >= 4)
        {
            shader->setVec3("points[0]", points[0]);
            shader->setVec3("points[1]", points[1]);
            shader->setVec3("points[2]", points[2]);
            shader->setVec3("points[3]", points[3]);
            glDrawArraysInstanced(GL_POINTS, 0, 1, discretization);

            glm::vec3 last_point = points[3];

            for(uint32_t i = 0; i < (points.size() - 4)/3; ++i)
            {
                shader->setVec3("points[0]", last_point);
                shader->setVec3("points[1]", points[3*(i+1)+1]);
                shader->setVec3("points[2]", points[3*(i+1)+2]);
                shader->setVec3("points[3]", points[3*(i+1)+3]);
                last_point = points[3*(i+1)+3];
                glDrawArraysInstanced(GL_POINTS, 0, 1, discretization);
            }
        }

    }

    virtual void onImGuiRender() override
    {
        ImGui::BeginMainMenuBar();

        if(ImGui::BeginMenu("Exercise"))
        {
            ImGui::MenuItem("Settings", nullptr, &show_test_window);
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();

        if(show_test_window)
        {
            ImGui::Begin("Settings", &show_test_window);
            ImGui::End();
        }

    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event& event) override
    {
        atcg::EventDispatcher distpatcher(event);
        if(atcg::Input::isKeyPressed(GLFW_KEY_LEFT_SHIFT))
        {
            distpatcher.dispatch<atcg::MouseButtonPressedEvent>(ATCG_BIND_EVENT_FN(G01Layer::onMousePressed));
        }
    }

    bool onMousePressed(atcg::MouseButtonPressedEvent& e)
    {
        if(points.size() >= 100)
        {
            printf("Max numbers of points reached\n");
            return false;
        }

        const auto& window = atcg::Application::get()->getWindow();

        float width = (float)window->getWidth();
        float height = (float)window->getHeight();
        glm::vec2 mouse_pos = atcg::Input::getMousePosition();

        float x_ndc = (static_cast<float>(mouse_pos.x)) / (static_cast<float>(width) / 2.0f) - 1.0f;
		float y_ndc = (static_cast<float>(height) - static_cast<float>(mouse_pos.y)) / (static_cast<float>(height) / 2.0f) - 1.0f;

		glm::vec3 world_pos(x_ndc, y_ndc, 0.0f);

        points.push_back(world_pos);
        indices.push_back(static_cast<uint32_t>(indices.size()));

        vbo->setData(reinterpret_cast<float*>(points.data()), static_cast<uint32_t>(points.size() * 3 * sizeof(float)));
        ibo->setData(indices.data(), static_cast<uint32_t>(indices.size()));
        return true;
    }

private:
    std::vector<glm::vec3> points;
    std::vector<uint32_t> indices;
    std::shared_ptr<atcg::VertexArray> vao;
    std::shared_ptr<atcg::VertexBuffer> vbo;
    std::shared_ptr<atcg::IndexBuffer> ibo;
    bool show_test_window = false;
};

class G01 : public atcg::Application
{
    public:

    G01()
        :atcg::Application()
    {
        pushLayer(new G01Layer("Layer"));
    }

    ~G01() {}

};

atcg::Application* atcg::createApplication()
{
    return new G01;
}