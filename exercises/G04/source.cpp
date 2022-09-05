#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <glfw/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <queue>

#include <numeric>

using VertexHandle = atcg::TriMesh::VertexHandle;

//Global mesh variable because we need it for the comparator
std::shared_ptr<atcg::TriMesh> mesh;
OpenMesh::VPropHandleT<double> property_distance;

class DistanceComparator {
public:
	bool operator() (const VertexHandle vh1, const VertexHandle vh2) {
		return mesh->property(property_distance, vh1) > mesh->property(property_distance, vh2);
	}
};


class G04Layer : public atcg::Layer
{
public:

    G04Layer(const std::string& name) : atcg::Layer(name) {}

    bool was_visited(VertexHandle vh) {
	// unvisted vertices are initialized with an infinite distance
	    return std::isfinite(mesh->property(property_distance, vh));
    }

    double distance_on_mesh(const std::shared_ptr<atcg::TriMesh>& mesh, const VertexHandle& vh_start, const VertexHandle& vh_target)
    {
        double distance = std::numeric_limits<double>::infinity();

        // Initialize the property with infinity to mark unvisited vertices
        for(auto vh: mesh->vertices()) {
            mesh->property(property_distance, vh) = std::numeric_limits<double>::infinity();
        }

        // The distance from the start vertex to itself is 0
        mesh->property(property_distance, vh_start) = 0.0;

        // A priority queue, which sorts the vertices by the distance to the source vertex
        std::priority_queue<VertexHandle, std::vector<VertexHandle>, DistanceComparator> queue;
        queue.push(vh_start);

        // Process vertices as long as the queue is not empty and the target is not found
        while(!queue.empty()) {
            // TODO: Implement the loop body
            // Hints:
            // - Loop invariant: all vertices in the queue have a valid distance, which is sum of the lengths
            //   of the edges on the shortest path between source_vh and the vertex.
            // - The top vertex in the priority queue always has the shortest path from the source vertex of all
            //   visited vertices.
            // - All vertices that need to be inserted into the queue at the end of the loop were not visisted before.
            // - Break the loop when the target vertex is found and assign its distance value to the "distance" variable.
            // - The only case when the loop is terminated because of an empty queue is when the target vertex cannot be reached.
            // - You can either use the Mesh::Point::norm method for distance calculations or the Mesh::calc_edge_length method,
            //   depending on which type of iterators you use for finding neighboring vertices.

            ////////////////////////////////////////////////// Example solution
            auto vh = queue.top();
            queue.pop();
            double my_distance = mesh->property(property_distance, vh);
            if(vh == vh_target) {
                distance = my_distance;
                break;
            } else {
                for(auto vit = mesh->cvv_ccwbegin(vh); vit != mesh->cvv_ccwend(vh); vit++) {
                    if(!was_visited(*vit)) {
                        mesh->property(property_distance, *vit) = my_distance + (mesh->point(vh) - mesh->point(*vit)).norm();
                        queue.push(*vit);
                    }
                }
            }
            ////////////////////////////////////////////////// End example solution
        }

        return distance;
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        mesh = std::make_shared<atcg::TriMesh>();
        OpenMesh::IO::read_mesh(*mesh.get(), "res/bunny.obj");

        //Each vertex now holds a distance property (double)
        mesh->add_property(property_distance);

        mesh->request_vertex_colors();
        for(auto v_it : mesh->vertices())
        {
            mesh->set_color(v_it, atcg::TriMesh::Color{0,0,0});
        }

        source_vh = mesh->vertex_handle(0);
        target_vh = mesh->vertex_handle(4200);

        render_mesh = std::make_shared<atcg::RenderMesh>();
        render_mesh->uploadData(mesh);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        if(render_mesh && render_faces)
            atcg::Renderer::draw(render_mesh, atcg::ShaderManager::getShader("base"), camera_controller->getCamera());

        if(render_mesh && render_points)
            atcg::Renderer::drawPoints(render_mesh, glm::vec3(0), atcg::ShaderManager::getShader("flat"), camera_controller->getCamera());

        if(render_mesh && render_edges)
            atcg::Renderer::drawLines(render_mesh, glm::vec3(0), camera_controller->getCamera());
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
            ImGui::MenuItem("Show Dijkstra Settings", nullptr, &show_dijkstra_settings);
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();

        if(show_dijkstra_settings)
        {
            ImGui::Begin("Settings Dijkstra", &show_dijkstra_settings);

            if(ImGui::Button("Calculate distance"))
            {
                final_distance = distance_on_mesh(mesh, source_vh, target_vh);
                clicked = true;
            }

            if(clicked)
            {
                ImGui::Text(("Distance between vertices: " + std::to_string(final_distance)).c_str());

                if(3.10 > final_distance || final_distance > 3.11)
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1,0,0,1));
                    ImGui::Text("Wrong distance!");
                    ImGui::PopStyleColor();
                }
            }

            ImGui::End();
        }

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
        dispatcher.dispatch<atcg::FileDroppedEvent>(ATCG_BIND_EVENT_FN(G04Layer::onFileDropped));
    }

    bool onFileDropped(atcg::FileDroppedEvent& event)
    {
        mesh = std::make_shared<atcg::TriMesh>();
        OpenMesh::IO::read_mesh(*mesh.get(), event.getPath());

        //Each vertex now holds a distance property (double)
        mesh->add_property(property_distance);

        mesh->request_vertex_colors();
        for(auto v_it : mesh->vertices())
        {
            mesh->set_color(v_it, atcg::TriMesh::Color{0,0,0});
        }

        source_vh = mesh->vertex_handle(0);
        target_vh = mesh->vertex_handle(4200);

        render_mesh = std::make_shared<atcg::RenderMesh>();
        render_mesh->uploadData(mesh);

        //Also reset camera
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        return true;
    }

private:
    std::shared_ptr<atcg::RenderMesh> render_mesh;
    std::shared_ptr<atcg::CameraController> camera_controller;

    bool show_dijkstra_settings = true;
    bool show_render_settings = false;
    bool render_faces = true;
    bool render_points = false;
    bool render_edges = false;
    double final_distance;
    bool clicked = false;

    VertexHandle source_vh;
    VertexHandle target_vh;
};

class G04 : public atcg::Application
{
    public:

    G04()
        :atcg::Application()
    {
        pushLayer(new G04Layer("Layer"));
    }

    ~G04() {}

};

atcg::Application* atcg::createApplication()
{
    return new G04;
}