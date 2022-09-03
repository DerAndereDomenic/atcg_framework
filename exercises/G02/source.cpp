#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <glfw/glfw3.h>
#include <imgui.h>

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
        for(uint32_t i = 0; i < grid->num_voxels(); ++i)
        {
            glm::ivec3 voxel = grid->index2voxel(i);
            glm::vec3 p = grid->voxel2position(voxel);

            (*grid)[i].sdf = sdf_heart(p);
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
        grid = std::make_shared<SDFGrid>(glm::vec3(0), 10, 0.1f);

        //Fill the grid with the sdf
        fillGrid(grid);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();
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