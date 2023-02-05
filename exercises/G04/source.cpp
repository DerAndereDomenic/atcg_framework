#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <queue>

#include <numeric>

using VertexHandle = atcg::Mesh::VertexHandle;

// Global mesh variable because we need it for the comparator
std::shared_ptr<atcg::Mesh> mesh;
OpenMesh::VPropHandleT<double> property_distance;

class DistanceComparator
{
public:
    bool operator()(const VertexHandle vh1, const VertexHandle vh2)
    {
        return mesh->property(property_distance, vh1) > mesh->property(property_distance, vh2);
    }
};


class G04Layer : public atcg::Layer
{
public:
    G04Layer(const std::string& name) : atcg::Layer(name) {}

    bool was_visited(VertexHandle vh)
    {
        // unvisted vertices are initialized with an infinite distance
        return std::isfinite(mesh->property(property_distance, vh));
    }

    double distance_on_mesh(const std::shared_ptr<atcg::Mesh>& mesh,
                            const VertexHandle& vh_start,
                            const VertexHandle& vh_target)
    {
        double distance = std::numeric_limits<double>::infinity();

        // Initialize the property with infinity to mark unvisited vertices
        for(auto vh: mesh->vertices())
        {
            mesh->property(property_distance, vh) = std::numeric_limits<double>::infinity();
        }

        // The distance from the start vertex to itself is 0
        mesh->property(property_distance, vh_start) = 0.0;

        // A priority queue, which sorts the vertices by the distance to the source vertex
        std::priority_queue<VertexHandle, std::vector<VertexHandle>, DistanceComparator> queue;
        queue.push(vh_start);

        // Process vertices as long as the queue is not empty and the target is not found
        while(!queue.empty())
        {
            // TODO: Implement the loop body
            // Hints:
            // - Loop invariant: all vertices in the queue have a valid distance, which is sum of the lengths
            //   of the edges on the shortest path between source_vh and the vertex.
            // - The top vertex in the priority queue always has the shortest path from the source vertex of all
            //   visited vertices.
            // - All vertices that need to be inserted into the queue at the end of the loop were not visisted before.
            // - Break the loop when the target vertex is found and assign its distance value to the "distance"
            // variable.
            // - The only case when the loop is terminated because of an empty queue is when the target vertex cannot be
            // reached.
            // - You can either use the Mesh::Point::norm method for distance calculations or the Mesh::calc_edge_length
            // method,
            //   depending on which type of iterators you use for finding neighboring vertices.

            ////////////////////////////////////////////////// Example solution
            auto vh = queue.top();
            queue.pop();
            double my_distance = mesh->property(property_distance, vh);
            if(vh == vh_target)
            {
                distance = my_distance;
                break;
            }
            else
            {
                for(auto vit = mesh->cvv_ccwbegin(vh); vit != mesh->cvv_ccwend(vh); vit++)
                {
                    if(!was_visited(*vit))
                    {
                        mesh->property(property_distance, *vit) =
                            my_distance + (mesh->point(vh) - mesh->point(*vit)).norm();
                        queue.push(*vit);
                    }
                }
            }
            ////////////////////////////////////////////////// End example solution
        }

        return distance;
    }

    //// Excercise 3: Use the distance values to trace a path from the target vertex back to the source vertex ////
    std::vector<VertexHandle> trace_back(const std::shared_ptr<atcg::Mesh>& mesh, const VertexHandle target_vh)
    {
        // Attention: This function only works after distance_on_mesh was executed with the same target_vh.

        std::vector<VertexHandle> result_path;
        VertexHandle current_vh = target_vh;
        result_path.push_back(target_vh);

        // TODO: Implement this
        // Hints:
        // - Each vertex on the path has a property with its distance to the source.
        // - Each vertex in an optimal path from the target to the source has a smaller distance than the vertex before.
        // - The source vertex has distance 0.
        ////////////////////////////////////////////////// Example solution
        while(mesh->property(property_distance, current_vh) > 0)
        {
            double min_distance = std::numeric_limits<double>::infinity();
            VertexHandle min_vh;
            for(auto vit = mesh->cvv_ccwbegin(current_vh); vit != mesh->cvv_ccwend(current_vh); vit++)
            {
                double my_distance = mesh->property(property_distance, *vit);
                if(my_distance < min_distance)
                {
                    min_distance = my_distance;
                    min_vh       = *vit;
                }
            }
            current_vh = min_vh;
            result_path.push_back(current_vh);
        }
        ////////////////////////////////////////////////// End example solution
        assert(mesh->property(property_distance, result_path.back()) == 0);

        std::reverse(result_path.begin(), result_path.end());
        return result_path;
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = std::make_shared<atcg::CameraController>(aspect_ratio);

        mesh = atcg::IO::read_mesh("res/bunny.obj");

        // Each vertex now holds a distance property (double)
        mesh->add_property(property_distance);

        mesh->request_vertex_colors();
        for(auto v_it: mesh->vertices()) { mesh->set_color(v_it, atcg::Mesh::Color {255, 255, 255}); }

        source_vh = mesh->vertex_handle(0);
        target_vh = mesh->vertex_handle(4200);

        mesh->uploadData();
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        if(mesh && render_faces)
            atcg::Renderer::draw(mesh, atcg::ShaderManager::getShader("base"), camera_controller->getCamera());

        if(mesh && render_points)
            atcg::Renderer::drawPoints(mesh,
                                       glm::vec3(0),
                                       atcg::ShaderManager::getShader("base"),
                                       camera_controller->getCamera());

        if(mesh && render_edges) atcg::Renderer::drawLines(mesh, glm::vec3(0), camera_controller->getCamera());
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
                final_distance   = distance_on_mesh(mesh, source_vh, target_vh);
                clicked_distance = true;
            }

            if(ImGui::Button("Dijkstra"))
            {
                distance_on_mesh(mesh, source_vh, target_vh);
                path             = trace_back(mesh, target_vh);
                clicked_dijkstra = true;

                if(path.size() != 113)
                    error_msg << "Wrong path length";
                else if(path.front() != source_vh || path.back() != target_vh)
                    error_msg << "Wrong source or target";
                // else
                {
                    // Color in vertices
                    for(auto vh: path) { mesh->set_color(vh, atcg::Mesh::Color {255, 0, 0}); }
                    mesh->uploadData();
                }
            }

            if(clicked_distance)
            {
                ImGui::Text(("Distance between vertices: " + std::to_string(final_distance)).c_str());

                if(3.10 > final_distance || final_distance > 3.11)
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0, 0, 1));
                    ImGui::Text("Wrong distance!");
                    ImGui::PopStyleColor();
                }
            }

            if(clicked_dijkstra)
            {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0, 0, 1));
                ImGui::Text(error_msg.str().c_str());
                ImGui::PopStyleColor();
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
        mesh = atcg::IO::read_mesh(event.getPath().c_str());

        // Each vertex now holds a distance property (double)
        mesh->add_property(property_distance);

        mesh->request_vertex_colors();
        for(auto v_it: mesh->vertices()) { mesh->set_color(v_it, atcg::Mesh::Color {255, 255, 255}); }

        source_vh = mesh->vertex_handle(0);
        target_vh = mesh->vertex_handle(4200);

        mesh->uploadData();

        // Also reset camera
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = std::make_shared<atcg::CameraController>(aspect_ratio);

        return true;
    }

private:
    std::shared_ptr<atcg::CameraController> camera_controller;

    bool show_dijkstra_settings = true;
    bool show_render_settings   = false;
    bool render_faces           = true;
    bool render_points          = false;
    bool render_edges           = false;
    double final_distance;
    bool clicked_distance = false;
    bool clicked_dijkstra = false;
    std::vector<VertexHandle> path;
    std::stringstream error_msg;

    VertexHandle source_vh;
    VertexHandle target_vh;
};

class G04 : public atcg::Application
{
public:
    G04() : atcg::Application() { pushLayer(new G04Layer("Layer")); }

    ~G04() {}
};

atcg::Application* atcg::createApplication()
{
    return new G04;
}