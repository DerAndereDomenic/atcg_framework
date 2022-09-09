# ATCG FRAMEWORK
This repository contains the practical exercises for the module "Advanced topics in Computer Graphics II".

## Building
First, you need to install a C++ compiler. Tested compilers that should work are MSVC for Windows and gcc for Linux. Clang or any Mac build is currently untested. Additionally, you need to install [CMake](https://cmake.org) to build the project.

To build the project clone the repository recursively (to include submodules)
```
git clone --recursive git@github.com:DerAndereDomenic/atcg_framework.git
```
If you already cloned the repository without recursive cloning, run
```
git submodule update --init --recursive
```
Navigate to the project folder and create and run the following commands to setup the project.

```
mkdir build
cd build
cmake ..
cd ..
```
From the main folder of this project (aka the folder where this file is located), run
```
cmake --build build --config <Debug|Release>
```
to compile the project. On Windows, you can specify if you want to build in Debug or Release mode. If everything worked the executables are located in ```bin\<config>\<name>.exe``` (or slightly differently depending on your system). Currently, you have to execute the program from the project's main folder. Otherwise, it may not find the shader files.

## Project Structure
The code base is based on the [Hazel Game Engine](https://github.com/TheCherno/Hazel). The framework also includes several dependencies that are used to implement the algorithms. You should make yourself familiar with **OpenMesh** explicitely. The project consists of the following components.

-**atcg_lib**: This library handles the rendering and event handling of the application. It defines an entry point for each application that uses this library. Each exercise uses this entry point to build its application.

-**exercises**: Contains the projects for each exercise. See _How to use_ for more details on its structure.

-**shader**: Contains the opengl shaders used for rendering. You can add custom shaders by providing a vertex (`<name>.vs`), fragment (`<name>.fs`), and (optionally) a geometry (`<name>.gs`) shader. To use them in a project you have to add it via
```c++
atcg::ShaderManager::addShader("<name>");
```
and use it by 
```c++
atcg::ShaderManager::getShader("<name>");
```
You can edit shaders while the program is running!

-**res**: Contains resources (meshes) used for the exercises

-**external**: Contains the exernal libraries used in this framework. There is no need to install any external libraries (except for OpenGL in some cases) as all dependencies come with this repository.

## Usage
All exercises have the same structure, that is rougly outlined here:
```c++
#include <Core/EntryPoint.h>
#include <ATCG.h>

//This class holds the methods that are called by the framework internally
class ExerciseLayer : public atcg::Layer
{
public:

    ExerciseLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        //Initialize members, etc.
        mesh = std::make_shared<atcg::Mesh>();
        OpenMesh::IO::read_mesh(*mesh.get(); "path/to/mesh");

        //...

        //Upload data onto GPU for rendering
        mesh->uploadData();
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        //Any physics based updates and rendering is handled here

        //...
        atcg::Renderer::draw(mesh, ShaderManager::getShader("base"), camera_controller->getCamera());
    }

    //All draw calls to ImGui to create a user interface
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

            ImGui::Checkbox("Render Vertices", &render_points);
            ImGui::Checkbox("Render Edges", &render_edges);
            ImGui::Checkbox("Render Mesh", &render_faces);
            ImGui::End();
        }

    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event& event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);

        //Add a custom function that gets called when an event is triggered
        dispatcher.dispatch<atcg::MouseClickedEvent>(ATCG_BIND_EVENT_FN(ExerciseLayer::onMouseClicked));
    }

    bool onMouseClicked(atcg::MouseClickedEvent& event)
    {
        //Do something if the mouse is clicked
    }

private:
    std::shared_ptr<atcg::CameraController> camera_controller;
    std::shared_ptr<atcg::Mesh> mesh;

    bool show_render_settings = false;
};

class Exercise : public atcg::Application
{
    public:

    Exercise()
        :atcg::Application()
    {
        pushLayer(new ExerciseLayer("Layer"));
    }

    ~Exercise() {}

};

//Entry point of the app
atcg::Application* atcg::createApplication()
{
    return new Exercise;
}
```

## Dependencies (included)
- [OpenMesh](https://gitlab.vci.rwth-aachen.de:9000/OpenMesh/OpenMesh) - For mesh manipulation.
- [Eigen](https://gitlab.com/libeigen/eigen/) - For math operations
- [Glad](https://glad.dav1d.de) - For loading OpenGL.
- [ImGui](https://github.com/ocornut/imgui) - For interactive GUI.
- [GLFW](https://github.com/glfw/glfw) - For window creation and handling
- [GLM](https://github.com/g-truc/glm) - For OpenGL math.