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

        vbo = std::make_shared<atcg::VertexBuffer>(static_cast<uint32_t>(max_num_points * sizeof(float) * 3));
        vbo->setLayout({{atcg::ShaderDataType::Float3, "aPosition"}});
        vao->addVertexBuffer(vbo);

        ibo = std::make_shared<atcg::IndexBuffer>(max_num_points);
        vao->setIndexBuffer(ibo);

        float aspect_ratio = (float)atcg::Application::get()->getWindow()->getWidth() / (float)atcg::Application::get()->getWindow()->getHeight();

        atcg::ShaderManager::addShaderFromName("bezier");
        atcg::ShaderManager::addShaderFromName("hermite");
    }

    void drawCurve(const std::shared_ptr<atcg::Shader>& shader, const glm::vec3& color)
    {
        int discretization = 20;
        shader->use();
        vao->use();
        shader->setInt("discretization", discretization);
        shader->setVec3("flat_color", color);

        if(points.size() > 0)
        {
            shader->setVec3("points[0]", points[0]);
            int32_t index = 1;
            for(uint32_t i = 1; i < points.size(); ++i)
            {
                shader->setVec3("points[" + std::to_string(index) + "]", points[i]);

                if(index == 3)
                {
                    glDrawArraysInstanced(GL_POINTS, 0, 1, discretization);
                    shader->setVec3("points[0]", points[i]);
                    index = 0;
                }
                ++index;
            }

            if(index != 1)
            {
                for(uint32_t i = index; i < 4; ++i)
                {
                    shader->setVec3("points["+ std::to_string(i) + "]", points.back());
                }
                glDrawArraysInstanced(GL_POINTS, 0, 1, discretization);
            }
        }
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        atcg::Renderer::clear();

        atcg::Renderer::drawPoints(vao, glm::vec3(0), atcg::ShaderManager::getShader("flat"));

        drawCurve(atcg::ShaderManager::getShader("bezier"), glm::vec3(1,0,0));
        drawCurve(atcg::ShaderManager::getShader("hermite"), glm::vec3(0,1,0));
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
        if(points.size() >= max_num_points)
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


        /*if(continuity_index == 3)
        {
            glm::vec3 second_last = points[points.size() - 2];

            glm::vec3 deriv = second_last - world_pos;

            points.push_back(world_pos - deriv);

            continuity_index = 1;
        }
        ++continuity_index;*/

        vbo->setData(reinterpret_cast<float*>(points.data()), static_cast<uint32_t>(points.size() * 3 * sizeof(float)));
        size_t old_size = indices.size();
        indices.resize(points.size());
        std::iota(indices.begin() + old_size, indices.end(), old_size);

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
    uint32_t max_num_points = 100;
    uint32_t continuity_index = 0;
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