#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

#include <glfw/glfw3.h>
#include <imgui.h>

class G01Layer : public atcg::Layer
{
public:

    G01Layer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;
        MyMesh mesh;
        OpenMesh::IO::read_mesh(mesh, "res/suzanne_blender.obj");

        mesh.request_vertex_normals();
        mesh.request_face_normals();
        mesh.update_normals();

        std::vector<float> position_data;
        std::vector<float> normal_data;
        position_data.resize(mesh.n_vertices() * 3);
        normal_data.resize(mesh.n_vertices() * 3);


        std::vector<uint32_t> indices_data;
        indices_data.resize(mesh.n_faces() * 3);

        for(auto vertex = mesh.vertices_begin(); vertex != mesh.vertices_end(); ++vertex)
        {
            int32_t vertex_id = vertex->idx();
            OpenMesh::Vec3f pos = mesh.point(*vertex);
            OpenMesh::Vec3f normal = mesh.calc_vertex_normal(*vertex);
            position_data[3 * vertex_id + 0] = pos[0];
            position_data[3 * vertex_id + 1] = pos[1];
            position_data[3 * vertex_id + 2] = pos[2];
            normal_data[3 * vertex_id + 0] = normal[0];
            normal_data[3 * vertex_id + 1] = normal[1];
            normal_data[3 * vertex_id + 2] = normal[2];
        }

        int32_t face_id = 0;
        for(auto face = mesh.faces_begin(); face != mesh.faces_end(); ++face)
        {
            int32_t vertex_id = 0;
            for(auto vertex = face->vertices().begin(); vertex != face->vertices().end(); ++vertex)
            {
                indices_data[3 * face_id + vertex_id] = vertex->idx();
                ++vertex_id;
            }
            ++face_id;
        }

        vao = std::make_shared<atcg::VertexArray>();
        std::shared_ptr<atcg::VertexBuffer> vbo_pos = std::make_shared<atcg::VertexBuffer>(position_data.data(), static_cast<uint32_t>(sizeof(float) * position_data.size()));
        vbo_pos->setLayout({
            {atcg::ShaderDataType::Float3, "aPosition"}
        });
        vao->addVertexBuffer(vbo_pos);

        std::shared_ptr<atcg::VertexBuffer> vbo_normal = std::make_shared<atcg::VertexBuffer>(normal_data.data(), static_cast<uint32_t>(sizeof(float) * normal_data.size()));
        vbo_normal->setLayout({
            {atcg::ShaderDataType::Float3, "aNormal"}
        });
        vao->addVertexBuffer(vbo_normal);

        std::shared_ptr<atcg::IndexBuffer> ibo = std::make_shared<atcg::IndexBuffer>(indices_data.data(), static_cast<uint32_t>(indices_data.size()));
        vao->setIndexBuffer(ibo);

        float aspect_ratio = (float)atcg::Application::get()->getWindow()->getWidth() / (float)atcg::Application::get()->getWindow()->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        atcg::Renderer::draw(vao, atcg::ShaderManager::getShader("base"), camera_controller->getCamera());

        if(atcg::Input::isKeyPressed(GLFW_KEY_SPACE))
        {
            std::cout << "Pressed Space!\n";
        }
    }

    virtual void onImGuiRender() override
    {
        ImGui::BeginMainMenuBar();

        if(ImGui::BeginMenu("Exercise"))
        {
            ImGui::MenuItem("Test", nullptr, &show_test_window);
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();

        if(show_test_window)
        {
            ImGui::Begin("Test", &show_test_window);
            ImGui::End();
        }

    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event& event) override
    {
        camera_controller->onEvent(event);
    }

private:
    std::shared_ptr<atcg::VertexArray> vao;
    std::shared_ptr<atcg::CameraController> camera_controller;
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