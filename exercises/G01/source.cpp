#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>

class G01Layer : public atcg::Layer
{
public:

    G01Layer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        //This data is used for rendering
        //The vbo holds the actual points to render
        //The vao holds information about the buffer structure. Here we only have a float3 for the position
        vao = std::make_shared<atcg::VertexArray>();

        vbo = std::make_shared<atcg::VertexBuffer>(static_cast<uint32_t>(max_num_points * sizeof(float) * 3));
        vbo->setLayout({{atcg::ShaderDataType::Float3, "aPosition"}});
        vao->addVertexBuffer(vbo);

        ibo = std::make_shared<atcg::IndexBuffer>(max_num_points);
        vao->setIndexBuffer(ibo);

        //Add custom shader for bezier and hermite curves
        atcg::ShaderManager::addShaderFromName("bezier");
        atcg::ShaderManager::addShaderFromName("hermite");

        const auto& window = atcg::Application::get()->getWindow();
        camera = std::make_shared<atcg::OrthographicCamera>(0, static_cast<float>(window->getWidth()), 0, static_cast<float>(window->getHeight()));
    }

    //This function renders the curves by passing the points as uniform data and using the supplied shader
    void drawCurve(const std::shared_ptr<atcg::Shader>& shader, const glm::vec3& color, const std::shared_ptr<atcg::Camera>& camera)
    {
        shader->use();
        vao->use();
        shader->setInt("discretization", discretization);
        shader->setVec3("flat_color", color);
        shader->setMVP(glm::mat4(1), camera->getView(), camera->getProjection());

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

            //Fill the last point into the uniforms if there were not enough points
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
        //Renders points and curves
        atcg::Renderer::clear();

        if(render_points && points.size() > 0)
            atcg::Renderer::drawPoints(vao, glm::vec3(0), atcg::ShaderManager::getShader("flat"), camera);

        if(render_polygon && points.size() > 0)
            atcg::Renderer::drawLines(vao, glm::vec3(0), atcg::ShaderManager::getShader("flat"), camera);

        if(render_bezier)
            drawCurve(atcg::ShaderManager::getShader("bezier"), glm::vec3(1,0,0), camera);

        if(render_hermite)
            drawCurve(atcg::ShaderManager::getShader("hermite"), glm::vec3(0,1,0), camera);
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
            if(ImGui::Button("Clear all"))
            {
                points.clear();
                indices.clear();
                continuity_index = 0;
            }

            ImGui::Checkbox("Render Points", &render_points);
            ImGui::Checkbox("Render Polygon", &render_polygon);
            ImGui::Checkbox("Render bezier", &render_bezier);
            ImGui::Checkbox("Render hermite", &render_hermite);
            ImGui::InputInt("Discretization", &discretization);
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
        distpatcher.dispatch<atcg::WindowResizeEvent>(ATCG_BIND_EVENT_FN(G01Layer::onWindowResized));
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

		glm::vec4 world_pos(x_ndc, y_ndc, 0.0f, 1.0f);

        world_pos = glm::inverse(camera->getViewProjection()) * world_pos;
        world_pos /= world_pos.w;

        points.push_back(world_pos);

        //We consider curves of degree 3. Therefore, to get a continuous spline we have to add
        //new points when we placed the third point. This happens hiere
        if(continuity_index == 3)
        {
            //TODO: Generate a new point to achieve C1-continuity
            //<solution>
            glm::vec3 second_last = points[points.size() - 2];

            glm::vec3 deriv = second_last - glm::vec3(world_pos);

            points.push_back(glm::vec3(world_pos) - deriv);
            //</solution>

            continuity_index = 1;
        }
        ++continuity_index;

        //Update the rendering data
        vbo->setData(reinterpret_cast<float*>(points.data()), static_cast<uint32_t>(points.size() * 3 * sizeof(float)));
        size_t old_size = indices.size();
        indices.resize(points.size());
        std::iota(indices.begin() + old_size, indices.end(), old_size);

        ibo->setData(indices.data(), static_cast<uint32_t>(indices.size()));
        return true;
    }

    bool onWindowResized(atcg::WindowResizeEvent& event)
    {
        camera->setProjection(0, static_cast<float>(event.getWidth()) , 0, static_cast<float>(event.getHeight()));
        return false;
    }

private:
    std::vector<glm::vec3> points;
    std::vector<uint32_t> indices;
    std::shared_ptr<atcg::VertexArray> vao;
    std::shared_ptr<atcg::VertexBuffer> vbo;
    std::shared_ptr<atcg::IndexBuffer> ibo;
    std::shared_ptr<atcg::OrthographicCamera> camera;
    bool show_test_window = false;
    uint32_t max_num_points = 100;
    uint32_t continuity_index = 0;
    bool render_bezier = true;
    bool render_hermite = true;
    bool render_points = true;
    bool render_polygon = true;
    int discretization = 20;
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