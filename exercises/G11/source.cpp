#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <queue>

#include <numeric>

class G11Layer : public atcg::Layer
{
public:

    G11Layer(const std::string& name) : atcg::Layer(name) {}

    template<typename T>
    struct LaplaceUniform
    {
        atcg::Laplacian<T> calculate(const std::shared_ptr<atcg::Mesh>& mesh)
        {
            std::vector<Eigen::Triplet<T>> edge_weights;

            for(auto e_it = mesh->edges_begin(); e_it != mesh->edges_end(); ++e_it)
            {
                uint32_t i = e_it->v0().idx();
                uint32_t j = e_it->v1().idx();

                edge_weights.emplace_back(i, j, T(1.0));
                edge_weights.emplace_back(j, i, T(1.0));
                edge_weights.emplace_back(i, i, T(-1.0));
                edge_weights.emplace_back(j, j, T(-1.0));
            }

            std::vector<Eigen::Triplet<T>> vertex_weights;

            for(auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end(); ++v_it)
            {
                vertex_weights.emplace_back(v_it->idx(), v_it->idx(), T(v_it->valence()));
            }

            size_t N = mesh->n_vertices();

            atcg::Laplacian<T> laplace;
            laplace.S.resize(N, N);
            laplace.M.resize(N, N);

            laplace.S.setFromTriplets(edge_weights.begin(), edge_weights.end());
            laplace.M.setFromTriplets(vertex_weights.begin(), vertex_weights.end());

            return laplace;
        }
    };

    std::vector<float> linspace(float a, float b, uint32_t steps)
    {
        float step_size = (b-a) / (steps - 1);

        std::vector<float> space(steps);

        for(uint32_t i = 0; i < steps; ++i)
        {
            space[i] = (a + i * step_size);
        }

        return space;
    }

    std::shared_ptr<atcg::Mesh> triangulate(const std::vector<atcg::Mesh::Point>& points)
    {
        std::shared_ptr<atcg::Mesh> mesh = std::make_shared<atcg::Mesh>();
        
        std::vector<atcg::Mesh::VertexHandle> v_handles(points.size());

        for(uint32_t i = 0; i < points.size(); ++i)
            v_handles[i] = mesh->add_vertex({points[i][0], points[i][2], points[i][1]});

        uint32_t grid_size = static_cast<uint32_t>(std::sqrt(points.size())) - 1;

        for(uint32_t grid_x = 0; grid_x < grid_size; ++grid_x)
        {
            for(uint32_t grid_y = 0; grid_y < grid_size; ++grid_y)
            {
                auto v00 = v_handles[grid_x + (grid_size+1) * grid_y];
                auto v10 = v_handles[grid_x + 1 + (grid_size+1) * grid_y];
                auto v01 = v_handles[grid_x + (grid_size+1) * (grid_y + 1)];
                auto v11 = v_handles[grid_x + 1 + (grid_size+1) * (grid_y + 1)];

                mesh->add_face(v00, v01, v10);
                mesh->add_face(v10, v01, v11);
            }
        }

        return mesh;
    }

    Eigen::SparseMatrix<float> slice_rows(const Eigen::SparseMatrix<float>& M, const std::vector<int>& indices)
    {
        std::vector<Eigen::Triplet<float>> vals;
        Eigen::SparseMatrix<float> Mt = M.transpose();
        int slice = 0;
        for(int index : indices)
        {
            for(Eigen::InnerIterator<Eigen::SparseMatrix<float>> it = Eigen::InnerIterator(Mt, index); it; ++it)
            {
                vals.emplace_back(slice, it.row(), it.value());
            }
            ++slice;
        }
                    
        
        Eigen::SparseMatrix<float> res;
        res.resize(indices.size(), M.cols());
        res.setFromTriplets(vals.begin(), vals.end());

        return res;
    }

    Eigen::SparseMatrix<float> slice_cols(const Eigen::SparseMatrix<float>& M, const std::vector<int>& indices)
    {
        std::vector<Eigen::Triplet<float>> vals;
        int slice = 0;
        for(int index : indices)
        {
            for(Eigen::InnerIterator<Eigen::SparseMatrix<float>> it = Eigen::InnerIterator(M, index); it; ++it)
            {
                vals.emplace_back(it.row(), slice, it.value());
            }
            ++slice;
        }
        
        Eigen::SparseMatrix<float> res;
        res.resize(M.rows(), indices.size());
        res.setFromTriplets(vals.begin(), vals.end());

        return res;
    }

    Eigen::MatrixXf openmesh2eigen(const std::vector<atcg::Mesh::Point>& points)
    {
        Eigen::MatrixXf M(points.size(), 3);

        for(uint32_t i = 0; i < points.size(); ++i)
        {
            M(i, 0) = points[i][0];
            M(i, 1) = points[i][1];
            M(i, 2) = points[i][2];
        }

        return M;
    }

    void updatePositions(const std::shared_ptr<atcg::Mesh>& mesh, const Eigen::MatrixXf& points)
    {
        for(uint32_t i = 0; i < points.rows(); ++i)
        {
            mesh->set_point(atcg::VertexHandle(i), {points(i,0), points(i,2), points(i,1)});
        }
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        std::vector<float> U = linspace(-1,1,100);
        std::vector<atcg::Mesh::Point> grid;

        for(float u : U)
        {
            for(float v : U)
            {
                grid.push_back({v, u, 0.f});
            }
        }

        mesh = triangulate(grid);
        atcg::Laplacian<float> laplacian = LaplaceUniform<float>().calculate(mesh);
        Eigen::SparseMatrix<float> L = laplacian.M.cwiseInverse() * laplacian.S;

        std::vector<int> boundary;
        std::vector<int> interior;
        for(auto v : mesh->vertices())
        {
            if(mesh->is_boundary(v))
            {
                boundary.push_back(v.idx());
            }
            else
            {
                interior.push_back(v.idx());
            }
        }

        Eigen::SparseMatrix<float> Lin = slice_rows(L, interior);
        Eigen::SparseMatrix<float> Linin = slice_cols(Lin, interior);
        Eigen::SparseMatrix<float> Linb = slice_cols(Lin, boundary);

        Eigen::MatrixXf z = openmesh2eigen(grid);

        Eigen::MatrixXf zb = z(boundary, Eigen::placeholders::all);

        for(int i = 0; i < zb.rows(); ++i)
        {
            float u = zb(i,0);
            float v = M_PI * zb(i,1);

            zb(i,0) = u * std::cos(v);
            zb(i,1) = u * std::sin(v);
            zb(i,2) = v;
        }

        Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
        solver.compute(Linin);
        Eigen::MatrixXf zi = solver.solve(-Linb*zb);

        z(interior, Eigen::placeholders::all) = zi;
        z(boundary, Eigen::placeholders::all) = zb;

        updatePositions(mesh, z);

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
            atcg::Renderer::drawPoints(mesh, glm::vec3(0), atcg::ShaderManager::getShader("base"), camera_controller->getCamera());

        if(mesh && render_edges)
            atcg::Renderer::drawLines(mesh, glm::vec3(1), camera_controller->getCamera());
    }

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
    }

private:
    std::shared_ptr<atcg::CameraController> camera_controller;
    std::shared_ptr<atcg::Mesh> mesh;

    bool show_render_settings = true;
    bool render_faces = true;
    bool render_points = false;
    bool render_edges = false;
};

class G11 : public atcg::Application
{
    public:

    G11()
        :atcg::Application()
    {
        pushLayer(new G11Layer("Layer"));
    }

    ~G11() {}

};

atcg::Application* atcg::createApplication()
{
    return new G11;
}