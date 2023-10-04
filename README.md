# ATCG FRAMEWORK

This repository contains a C++ rendering framework.

## Building

First, you need to install a C++ compiler. Tested compilers that should work are MSVC for Windows and gcc for Linux. MacOS is only tested via github actions. Additionally, you need to install [CMake](https://cmake.org) to build the project.

To build the project clone the repository recursively (to include submodules)

```
git clone --recursive git@github.com:DerAndereDomenic/atcg_framework.git
```

If you already cloned the repository without recursive cloning, run

```
git submodule update --init --recursive
```

Navigate to the project folder and create and run the following command to setup the project.

```
cmake . -B build
```

From the main folder of this project (aka the folder where this file is located), run

```
cmake --build build --config <Debug|Release>
```

to compile the project. On Windows, you can specify if you want to build in Debug or Release mode. If everything worked the executables are located in `bin\<config>\<name>.exe` (or slightly differently depending on your system). Currently, you have to execute the program from the project's main folder. Otherwise, it may not find the shader files.

### Building with CUDA Support

You can also build the framework with CUDA support by setting the ATCG_CUDA_BACKEND option

```
cmake . -B build -DATCG_CUDA_BACKEND=ON
```

This way you can include high performant GPU code into your application. Note that this is only possible if you have the CUDA compiler installed.

### Building the documentation (experimental)

There is a (not very complete) documentation that can be build using [doxygen](https://www.doxygen.nl/index.html) and [sphinx](https://www.sphinx-doc.org/en/master/). Install the dependencies listed in this [Blog Post](https://devblogs.microsoft.com/cppblog/clear-functional-c-documentation-with-sphinx-breathe-doxygen-cmake/) and set the ATCG_BUILD_DOCS option to true when generating the CMake project.

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
class MyLayer : public atcg::Layer
{
public:

    MyLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        //Initialize members, etc.
        // ...

        // Load meshes
        mesh = atcg::IO::read_mesh("path/to/mesh");

        // Load images and create textures
        diffuse_image = atcg::IO::imread("path/to/image");
        atcg::TextureSpecification spec;
        spec.width = diffuse_image->width();
        spec.height = diffuse_image->height();
        diffuse_texture = atcg::Texture2D::create(diffuse_image, spec);

        // Create a scene using the custom shared pointer atcg::ref_ptr
        scene = atcg::make_ref<atcg::Scene>();

        // Create an entity inside the scene
        atcg::Entity entity = scene->createEntity("Entity Name");

        // Add components to entity and edit them
        entity.addComponent<atcg::TransformComponent>();
        entity.addComponent<atcg::GeometryComponent>(mesh);
        auto& renderer = entity.addComponent<atcg::MeshRenderComponent>();
        renderer.material.setDiffuseTexture(diffuse_texture);
        renderer.material.setMetallic(0.5f);

        //...
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        //Any physics based updates and rendering is handled here

        //...
        atcg::Renderer::draw(scene, camera_controller->getCamera());
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
        dispatcher.dispatch<atcg::MouseClickedEvent>(ATCG_BIND_EVENT_FN(MyLayer::onMouseClicked));
    }

    bool onMouseClicked(atcg::MouseClickedEvent& event)
    {
        //Do something if the mouse is clicked
    }

private:
    atcg::ref_ptr<atcg::Scene> scene;
    atcg::ref_ptr<atcg::CameraController> camera_controller;
    atcg::ref_ptr<atcg::Mesh> mesh;

    atcg::ref_ptr<atcg::Image> diffuse_image;
    atcg::ref_ptr<atcg::Texture2D> diffuse_texture;

    bool show_render_settings = false;
};

class MyApplication : public atcg::Application
{
    public:

    MyApplication()
        :atcg::Application()
    {
        pushLayer(new MyLayer("Layer"));
    }

    ~MyApplication() {}

};

//Entry point of the app
atcg::Application* atcg::createApplication()
{
    return new MyApplication;
}
```

### Custom shared pointers

The framework internally uses a custom shared pointer implementation that is very similar to the STL shared_ptr architecture (although not all std::shared_ptr features are implemented):

```c++
atcg::ref_ptr<T> p = atcg::make_ref<T>(...);
p->foo();
T* raw = p.get();
```

This allows us to introduce host and device shared pointers if CUDA support is enabled. By using the device allocator

```c++
atcg::ref_ptr<T, atcg::device_allocator> p;
```

this shared pointer acts as a standard std::shared_ptr but handles CUDA device memory (i.e. the memory is freed if the internal references count reaches zero).

IMPORTANT: Note that CUDAs memory API is more a C than C++ API. Therefore object construction works a bit differently. Do NOT use atcg::make_shared when constructing a device ptr but use the given constructors. To initialize an object correctly, you have to copy it from host.

```c++
// Creates a device buffer with 10 ints
atcg::ref_ptr<int, atcg::device_allocator> p(10);

// ...

// Creates an object and initializes it from a host object
T host_object(...);
// atcg::dref_ptr as shortcut for atcg::ref_ptr<..., atcg::device_allocator>
atcg::dref_ptr<T> device_ptr(1); // Allocate space for one object
device_ptr.upload(&host_object);
```

## Dependencies (included)

- [OpenMesh](https://gitlab.vci.rwth-aachen.de:9000/OpenMesh/OpenMesh) - For mesh manipulation.
- [Eigen](https://gitlab.com/libeigen/eigen/) - For math operations
- [Glad](https://glad.dav1d.de) - For loading OpenGL.
- [ImGui](https://github.com/ocornut/imgui) - For interactive GUI.
- [GLFW](https://github.com/glfw/glfw) - For window creation and handling
- [GLM](https://github.com/g-truc/glm) - For OpenGL math.
- [entt](https://github.com/skypjack/entt) - For the entity component system.
- [ImGuizmo](https://github.com/CedricGuillemet/ImGuizmo) - For guizmos.
- [ImPlot](https://github.com/epezent/implot) - For plots.
- [nanoflann](https://github.com/jlblancoc/nanoflann) - For kD-Trees.
- [nanort](https://github.com/lighttransport/nanort) - For ray - mesh intersection tests.
- [protable-file-dialogs](https://github.com/samhocevar/portable-file-dialogs) - For file dialogs.
- [spdlog](https://github.com/gabime/spdlog) - For logging.
- [pybind11](https://github.com/pybind/pybind11) - For python bindings.
- [stbimage](https://github.com/nothings/stb) - For image I/O.
- [tinyobjload](https://github.com/tinyobjloader/tinyobjloader) - For loading obj meshes.
- [yaml-cpp](https://github.com/jbeder/yaml-cpp) - For serializing YAML files.
