#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <glfw/glfw3.h>
#include <imgui.h>

#include "MarchingCubesTable.h"

class G02Layer : public atcg::Layer
{
public:

    G02Layer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
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

    }

private:

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