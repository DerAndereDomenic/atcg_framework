#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <glfw/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <queue>

#include <numeric>

using VertexHandle = atcg::Mesh::VertexHandle;
using EdgeHandle = atcg::Mesh::EdgeHandle;

class G07Layer : public atcg::Layer
{
public:

    G07Layer(const std::string& name) : atcg::Layer(name) {}

    enum class WeightType
    {
        UNIFORM_SPRING,
        CHORDAL_SPRING,
        WACHSPRESS,
        DISCRETE_HARMONIC,
        MEAN_VALUE
    };

    std::vector<EdgeHandle> detect_boundary_edges(const std::shared_ptr<atcg::Mesh>& mesh)
    {
        std::vector<EdgeHandle> boundary_edges;

        for(auto e_it = mesh->edges_begin(); e_it != mesh->edges_end(); ++e_it)
        {
            if(mesh->is_boundary(*e_it))
            {
                boundary_edges.push_back(*e_it);
            }
        }

        return boundary_edges;
    }

    std::vector<VertexHandle> detect_boundary_path(const std::shared_ptr<atcg::Mesh>& mesh, const std::vector<EdgeHandle>& boundary_edges)
    {
        std::vector<VertexHandle> boundary_path;
        
        VertexHandle start = mesh->from_vertex_handle(mesh->halfedge_handle(boundary_edges[0], 0));
        VertexHandle current_to = mesh->to_vertex_handle(mesh->halfedge_handle(boundary_edges[0], 0));
        boundary_path.push_back(start);
        boundary_path.push_back(current_to);

        for(uint32_t i = 1; i < boundary_edges.size(); ++i)
        {
            for(uint32_t j = 1; j < boundary_edges.size(); ++j)
            {
                VertexHandle from = mesh->from_vertex_handle(mesh->halfedge_handle(boundary_edges[j], 0));
                if(from.idx() == current_to.idx())
                {
                    current_to = mesh->to_vertex_handle(mesh->halfedge_handle(boundary_edges[j], 0));
                    boundary_path.push_back(current_to);
                    break;
                }
            }
        }

        return boundary_path;
    }

    std::vector<float> path_length(const std::shared_ptr<atcg::Mesh>& mesh, const std::vector<VertexHandle>& path)
    {
        std::vector<float> path_lengths;

        for(uint32_t i = 0; i < path.size() - 1; ++i)
        {
            atcg::TriMesh::Point p0 = mesh->point(path[i]);
            atcg::TriMesh::Point p1 = mesh->point(path[i+1]);
            path_lengths.push_back((p0-p1).norm());
        }

        return path_lengths;
    }

    std::vector<atcg::Mesh::Point> map_boundary_edges_to_circle(const std::vector<float>& edge_lengths)
    {
        float total_length = std::accumulate(edge_lengths.begin(), edge_lengths.end(), 0.0f);

        std::vector<atcg::Mesh::Point> circle;
        circle.push_back({1.0f, 0.0f, 0.0f});
        float angle = 0;
        for(uint32_t i = 0; i < edge_lengths.size(); ++i)
        {
            angle += edge_lengths[i] / total_length * 2.0f * static_cast<float>(M_PI);
            circle.push_back(atcg::Mesh::Point{std::cos(angle), std::sin(angle), 0.0f});
        }
        return circle;
    }

    inline float angle_from_metric(float a, float b, float c)
    {
        /* numerically stable version of law of cosines
        * angle between a and b, opposite to edge c
        */

        float alpha = acos((a*a + b*b - c*c) / (2.0 * a * b));

        if (alpha < 1e-8f)
        {
            alpha = std::sqrt((c*c - (a - b)*(a - b)) / (2.0 * a * b));
            std::cout << "small angle < 1e-8!" << std::endl;
        }
        return alpha;
    }

    std::vector<Eigen::Triplet<float>> method_switcher(const std::shared_ptr<atcg::Mesh>& mesh, const WeightType& method)
    {
        std::vector<Eigen::Triplet<float>> coefficients;

        uint32_t n_faces = mesh->n_faces();

        for(auto f_it = mesh->faces_begin(); f_it != mesh->faces_end(); ++f_it)
        {
            std::vector<VertexHandle> v;
            for(auto v_it = f_it->vertices().begin(); v_it != f_it->vertices().end(); ++ v_it)
            {
                v.push_back(*v_it);
            }
            assert(v.size() == 3);

            atcg::Mesh::Point vi = mesh->point(v[0]);
            atcg::Mesh::Point vj = mesh->point(v[1]);
            atcg::Mesh::Point vk = mesh->point(v[2]);

            int i = v[0].idx();
            int j = v[1].idx();
            int k = v[2].idx();

            float rij = (vj - vi).norm();
            float rjk = (vk - vj).norm();
            float rki = (vi - vk).norm();

            float alphai = angle_from_metric(rij, rki, rjk);
            float alphaj = angle_from_metric(rjk, rij, rki);
            float alphak = angle_from_metric(rjk, rki, rij);

            float wij = 0, wjk = 0, wki = 0;
            float wji = 0, wkj = 0, wik = 0;

            switch(method)
            {
                case WeightType::UNIFORM_SPRING :
                {
                    // implement here uniform weights
                    //<solution>
                    wij = 1.0f / 2.0f;
                    wjk = 1.0f / 2.0f;
                    wki = 1.0f / 2.0f;

                    wji = wij;
                    wkj = wjk;
                    wik = wki;
                    //</solution>
                }break;

                case WeightType::CHORDAL_SPRING :
                {
                    //implement here chordal spring weitghts: w = 1.0 / r^2
                    //<solution>
                    wij = 1.0f / 2.0f * rij * rij;
                    wjk = 1.0f / 2.0f * rjk * rjk;
                    wki = 1.0f / 2.0f * rki * rki;

                    wji = wij;
                    wkj = wjk;
                    wik = wki;
                    //</solution>
                }break;

                case WeightType::WACHSPRESS :
                {
                    //implement here the wachspress weights
                    //<solution>
                    wij = 1.0f / std::tan(alphaj) / (rij * rij);
                    wkj = 1.0f / std::tan(alphaj) / (rjk * rjk);

                    wjk = 1.0f / std::tan(alphak) / (rjk * rjk);
                    wik = 1.0f / std::tan(alphak) / (rki * rki);

                    wki = 1.0f / std::tan(alphai) / (rki * rki);
                    wji = 1.0f / std::tan(alphai) / (rij * rij);
                    //</solution>
                }break;

                case WeightType::DISCRETE_HARMONIC :
                {
                    //implement here the discrete harmonic weights
                    //<solution>
                    wij = 1.0f / std::tan(alphak);
                    wjk = 1.0f / std::tan(alphai);
                    wki = 1.0f / std::tan(alphaj);

                    wji = wij;
                    wkj = wjk;
                    wik = wki;
                    //</solution>
                }break;

                case WeightType::MEAN_VALUE :
                {
                    //implement here the mean value weights
                    //<solution>
                    wij = std::tan(alphai / 2.0f) / rij;
                    wji = std::tan(alphaj / 2.0f) / rij;

                    wjk = std::tan(alphaj / 2.0f) / rjk;
                    wkj = std::tan(alphak / 2.0f) / rjk;

                    wki = std::tan(alphak / 2.0f) / rki;
                    wik = std::tan(alphai / 2.0f) / rki;
                    //</solution>
                }break;
            }

            coefficients.emplace_back(i, j, wij);
            coefficients.emplace_back(j, k, wjk);
            coefficients.emplace_back(k, i, wki);

            //symmetric part
            coefficients.emplace_back(j, i, wji);
            coefficients.emplace_back(k, j, wkj);
            coefficients.emplace_back(i, k, wik);
        }

        return coefficients;
    }

    Eigen::SparseMatrix<float> construct_operator(const std::shared_ptr<atcg::Mesh>& mesh, 
                                                  const std::vector<Eigen::Triplet<float>>& coefficients,
                                                  const std::vector<VertexHandle>& path)
    {
        Eigen::SparseMatrix<float> op(mesh->n_vertices(), mesh->n_vertices());
        op.setFromTriplets(coefficients.begin(), coefficients.end());

        Eigen::VectorXf row_sum = op * Eigen::VectorXf::Ones(op.cols());

        Eigen::SparseMatrix<float> Id(mesh->n_vertices(), mesh->n_vertices());
        Id.setIdentity();

        Eigen::SparseMatrix<float> W = Id * row_sum.asDiagonal();
        op = (W - op).eval();

        op = (W.cwiseInverse() * op).eval();

        Eigen::VectorXf ones = Eigen::VectorXf::Ones(mesh->n_vertices());
        Eigen::VectorXf zeros = Eigen::VectorXf::Zero(mesh->n_vertices());

        for(uint32_t i = 0; i < path.size(); ++i)
        {
            ones(path[i].idx()) = 0;
            zeros(path[i].idx()) = 1;
        }

        op = (Id * ones.asDiagonal() * op).eval();
        op = (op + Id * zeros.asDiagonal()).eval();

        return op;
    }

    Eigen::MatrixXf construct_rhs(const std::shared_ptr<atcg::Mesh>& mesh, 
                                  const std::vector<VertexHandle>& path, 
                                  const std::vector<atcg::Mesh::Point>& boundary_constraints)
    {
        Eigen::MatrixXf rhs = Eigen::MatrixXf::Zero(mesh->n_vertices(), 3);

        for(uint32_t i = 0; i < boundary_constraints.size(); ++i)
        {
            rhs(path[i].idx(), 0) = -boundary_constraints[i][0];
            rhs(path[i].idx(), 1) = -boundary_constraints[i][1];
            rhs(path[i].idx(), 2) = -boundary_constraints[i][2];
        }

        return rhs;
    }

    Eigen::MatrixXf solve(const Eigen::SparseMatrix<float>& op, const Eigen::MatrixXf& rhs)
    {
        Eigen::BiCGSTAB<Eigen::SparseMatrix<float>> solver;
        solver.compute(op);
        return solver.solve(rhs).eval();
    }

    Eigen::MatrixXf reparameterize(const std::shared_ptr<atcg::Mesh>& mesh, 
                        const std::vector<VertexHandle>& path, 
                        const std::vector<atcg::Mesh::Point>& constraints, 
                        const WeightType& method)
    {
        std::vector<Eigen::Triplet<float>> coefficients = method_switcher(mesh, method);

        Eigen::SparseMatrix<float> op = construct_operator(mesh, coefficients, path);

        Eigen::MatrixXf rhs = construct_rhs(mesh, path, constraints);

        Eigen::MatrixXf uv = solve(op, rhs);

        return uv;
    }

    void apply_parameterization(const std::shared_ptr<atcg::Mesh>& mesh, const Eigen::MatrixXf& uv)
    {
        for(auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end(); ++v_it)
        {
            atcg::Mesh::Point p{uv(v_it->idx(), 0), uv(v_it->idx(), 1), 0.0f};
            mesh->set_point(*v_it, p);
        }
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        mesh_original = std::make_shared<atcg::Mesh>();
        OpenMesh::IO::read_mesh(*mesh_original.get(), "res/maxear.obj");
        mesh_original->request_vertex_colors();

        mesh = std::make_shared<atcg::Mesh>(*mesh_original.get());

        float max_scale = -std::numeric_limits<float>::infinity();
        atcg::TriMesh::Point mean_point{0,0,0};
        for(auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end(); ++v_it)
        {
            for(uint32_t i = 0; i < 3; ++i)
            {
                if(mesh->point(*v_it)[i] > max_scale)
                {
                    max_scale = mesh->point(*v_it)[i];
                }
            }

            mean_point += mesh->point(*v_it);
        }
        mean_point /= static_cast<float>(mesh->n_vertices());

        for(auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end(); ++v_it)
        {
            mesh->set_point(*v_it, (mesh->point(*v_it) - mean_point) / max_scale);
        }

        boundary_edges = detect_boundary_edges(mesh_original);
        boundary_path = detect_boundary_path(mesh_original, boundary_edges);
        edge_lengths = path_length(mesh_original, boundary_path);
        circle = map_boundary_edges_to_circle(edge_lengths);

        Eigen::MatrixXf uv = reparameterize(mesh_original, boundary_path, circle, WeightType::UNIFORM_SPRING);

        apply_parameterization(mesh, uv);

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
            atcg::Renderer::drawLines(mesh, glm::vec3(0), camera_controller->getCamera());
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
    }

private:
    std::shared_ptr<atcg::CameraController> camera_controller;
    std::shared_ptr<atcg::Mesh> mesh;
    std::shared_ptr<atcg::Mesh> mesh_original;

    std::vector<EdgeHandle> boundary_edges;
    std::vector<VertexHandle> boundary_path;
    std::vector<float> edge_lengths;
    std::vector<atcg::Mesh::Point> circle;

    bool show_render_settings = false;
    bool render_faces = false;
    bool render_points = false;
    bool render_edges = true;
};

class G07 : public atcg::Application
{
    public:

    G07()
        :atcg::Application()
    {
        pushLayer(new G07Layer("Layer"));
    }

    ~G07() {}

};

atcg::Application* atcg::createApplication()
{
    return new G07;
}