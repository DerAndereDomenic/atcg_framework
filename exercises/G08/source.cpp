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
using EdgeHandle = atcg::Mesh::EdgeHandle;
using HalfEdgeHandle = atcg::Mesh::HalfedgeHandle;
using GeodesicDistanceProperty = OpenMesh::VPropHandleT<double>;

template<typename T>
struct LaplaceCotan
{
    T clampCotan(T v)
    {
        const T bound = 19.1;
        return (v < -bound ? -bound : (v > bound ? bound : v));
    }

    T triangleCotan(const atcg::TriMesh::Point& v0, const atcg::TriMesh::Point& v1, const atcg::TriMesh::Point& v2)
    {
        const auto d0 = v0 - v2;
        const auto d1 = v1 - v2;
        const auto d2 = v1 - v0;
        const auto area = atcg::areaFromMetric<T>(d0.norm(), d1.norm(), d2.norm());
        if(area > 1e-5)
            return clampCotan(d0.dot(d1) / area);
        return 1e-5;
    }

    atcg::Laplacian<T> calculate(const std::shared_ptr<atcg::Mesh>& mesh)
    {
        std::vector<Eigen::Triplet<T>> edge_weights;

        for(auto e_it = mesh->edges_begin(); e_it != mesh->edges_end(); ++e_it)
        {
            uint32_t i = e_it->v0().idx();
            uint32_t j = e_it->v1().idx();

            const auto h0 = e_it->h0();
            const auto h1 = e_it->h1();

            const auto p0 = h0.to();
            const auto p1 = h1.to();

            T weight = 0;
            if(!mesh->is_boundary(h0))
            {
                const auto p2 = h0.next().to();
                weight += triangleCotan(mesh->point(p0), mesh->point(p1), mesh->point(p2));
            }

            if(!mesh->is_boundary(h1))
            {
                const auto p2 = h1.next().to();
                weight += triangleCotan(mesh->point(p0), mesh->point(p1), mesh->point(p2));
            }

            edge_weights.emplace_back(i, j, weight);
            edge_weights.emplace_back(j, i, weight);
            edge_weights.emplace_back(i, i, -weight);
            edge_weights.emplace_back(j, j, -weight);
        }

        std::vector<Eigen::Triplet<T>> vertex_weights;

        for(auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end(); ++v_it)
        {
            T weight = 0;

            for(const auto& e : v_it->edges())
            {
                const auto h0 = e.h0();
                const auto h1 = e.h1();

                const auto p0 = h0.to();
                const auto p1 = h1.to();

                if(!mesh->is_boundary(h0))
                {
                    const auto p2 = h0.next().to();
                    weight += triangleCotan(mesh->point(p0), mesh->point(p1), mesh->point(p2));
                }

                if(!mesh->is_boundary(h1))
                {
                    const auto p2 = h1.next().to();
                    weight += triangleCotan(mesh->point(p0), mesh->point(p1), mesh->point(p2));
                }
            }
            vertex_weights.emplace_back(v_it->idx(), v_it->idx(), weight);
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

class G08Layer : public atcg::Layer
{
public:

    G08Layer(const std::string& name) : atcg::Layer(name) {}

    void colorize_mesh(const std::shared_ptr<atcg::Mesh>& mesh, const OpenMesh::VPropHandleT<double>& vertexProperty)
    {
        assert(mesh->has_vertex_colors());
        double max_value = -std::numeric_limits<double>::infinity();
        double min_value = std::numeric_limits<double>::infinity();
        for(auto vh : mesh->vertices()) 
        {
            double value = mesh->property(vertexProperty, vh);
            if(!std::isfinite(value)) continue;
            max_value = std::max(max_value, value);
            min_value = std::min(min_value, value);
        }
        std::cout << min_value << " " << max_value << std::endl;
        for(auto vh : mesh->vertices()) 
        {
            double value = mesh->property(vertexProperty, vh);
            if(std::isfinite(value)) 
            {
                mesh->set_color(vh, {255 * (value - min_value) / (max_value - min_value), 0, 0});
            } 
            else 
            {
                mesh->set_color(vh, {0, 0, 255});
            }
        }
    }

    void cosine_colorize_mesh(const std::shared_ptr<atcg::Mesh>& mesh, const OpenMesh::VPropHandleT<double>& vertexProperty, const double periods)
    {
        assert(mesh->has_vertex_colors());
        double max_value = -std::numeric_limits<double>::infinity();
        double min_value = std::numeric_limits<double>::infinity();
        for(auto vh : mesh->vertices()) 
        {
            double value = mesh->property(vertexProperty, vh);
            if(!std::isfinite(value)) continue;
            max_value = std::max(max_value, value);
            min_value = std::min(min_value, value);
        }
        
        for(auto vh : mesh->vertices()) 
        {
            double value = ((mesh->property(vertexProperty, vh) - min_value) / (max_value - min_value)) * 2 * M_PI * periods;
            if(std::isfinite(value)) 
            {
                mesh->set_color(vh, {static_cast<unsigned char>(value / (2 * M_PI * periods) * 255 * (1.0 + cos(value)) / 2.0), 0, 0});
            }
            else
            {
                mesh->set_color(vh, {0, 0, 255});
            }
        }
    }

    struct BoundaryCondition 
    {
		Eigen::Index index;
		double value;
		double diagonal_value = 1.0;
	};

    void apply_boundary_conditions(Eigen::SparseMatrix<double> &laplacian, Eigen::VectorXd &rhs, const std::vector<std::pair<Eigen::Index, double>> &boundary_conditions) 
    {
        std::vector<BoundaryCondition> bcs;
        bcs.reserve(boundary_conditions.size());
        std::transform(boundary_conditions.begin(), boundary_conditions.end(), std::back_inserter(bcs), [&](auto bc) { return BoundaryCondition{bc.first, bc.second}; });
        apply_boundary_conditions(laplacian, rhs, bcs);
    }

    void apply_boundary_conditions(Eigen::SparseMatrix<double> &laplacian, Eigen::VectorXd &rhs, const std::vector<BoundaryCondition> &boundary_conditions) {
        std::set<Eigen::Index> boundary_indices;
        for(auto bc : boundary_conditions) 
        {
            boundary_indices.insert(bc.index);
            rhs -= laplacian.col(bc.index) * bc.value;
        }
        for(auto bc : boundary_conditions) 
        {
            rhs[bc.index] = bc.value;
        }
        for(int k = 0; k < laplacian.outerSize(); ++k) 
        {
            for(Eigen::SparseMatrix<double>::InnerIterator it(laplacian, k); it; ++it) 
            {
                if(boundary_indices.find(it.col()) != boundary_indices.end() || boundary_indices.find(it.row()) != boundary_indices.end()) 
                {
                    if(it.row() == it.col()) 
                    {
                        // handled below
                        // it.valueRef() = 1.0;
                    } 
                    else 
                    {
                        it.valueRef() = 0.0;
                    }
                }
            }
        }
        for(auto bc : boundary_conditions)
        {
            laplacian.coeffRef(bc.index, bc.index) = bc.diagonal_value;
        }
    }

    void compute_heat_geodesics(const std::shared_ptr<atcg::Mesh>& mesh, 
                                const std::vector<VertexHandle>& start_vhs,
                                GeodesicDistanceProperty& distance_property,
                                double t)
    {
        if(start_vhs.empty())return;

        mesh->request_face_normals();
        mesh->update_face_normals();

        atcg::Laplacian<double> laplacian = LaplaceCotan<double>().calculate(mesh);
        Eigen::SparseMatrix<double> Lc = laplacian.S;

        Eigen::VectorXd u0 = Eigen::VectorXd::Zero(mesh->n_vertices());
        for(auto start_vh : start_vhs)
        {
            u0[start_vh.idx()] = 1.0;
        }

        Eigen::VectorXd vertex_areas(mesh->n_vertices());
        vertex_areas.setZero();
        for(auto vh : mesh->vertices())
        {
            vertex_areas[vh.idx()] = mesh->area(vh);
        }

        Eigen::SparseMatrix<double> A_tLc = -t * Lc;
        assert(A_tLc.rows() == A_tLc.cols());
        for(uint32_t i = 0; i < A_tLc.rows(); ++i)
        {
            A_tLc.coeffRef(i, i) += vertex_areas[i];
        }
        assert(A_tLc.isCompressed());

        Eigen::SparseLU<Eigen::SparseMatrix<double>> luSolver;
        luSolver.compute(A_tLc);
        Eigen::VectorXd u = luSolver.solve(u0);

        std::vector<OpenMesh::Vec3d> face_grad_u(mesh->n_faces(), OpenMesh::Vec3d(0,0,0));
        for(auto fh : mesh->faces())
        {
            OpenMesh::Vec3d& x = face_grad_u[fh.idx()];
            const atcg::Mesh::Normal& N = mesh->normal(fh);
            for(auto v_it = mesh->cfv_ccwbegin(fh); v_it != mesh->cfv_ccwend(fh); ++v_it)
            {
                auto vh = *v_it;
                auto heh = mesh->opposite_halfedge_handle(fh, vh);
                atcg::Mesh::Point ei = mesh->point(mesh->to_vertex_handle(heh)) - mesh->point(mesh->from_vertex_handle(heh));
                x += u[v_it->idx()] * (N % ei);
            }
            x /= 2.0 * mesh->area(fh);
        }

        Eigen::VectorXd vertex_div_u = Eigen::VectorXd::Zero(mesh->n_vertices());
        for(auto vh : mesh->vertices()) 
        {
            auto pi = mesh->point(vh);
            double &div = vertex_div_u[vh.idx()];
            for(auto h_it = mesh->cvoh_ccwbegin(vh); h_it != mesh->cvoh_ccwend(vh); ++h_it) 
            {
                // The edge and the next edge belong to a common face,
                // except when the current edge is a boundary edge. In that case
                // we skip it, as it will be the next_heh of another halfedge later.
                auto heh = *h_it;
                if(mesh->is_boundary(heh)) continue;
                auto next_h_it = h_it;
                next_h_it++;
                auto next_heh = *next_h_it;

                auto fh = mesh->face_handle(heh);

                auto p1 = mesh->point(mesh->to_vertex_handle(heh));
                auto p2 = mesh->point(mesh->to_vertex_handle(next_heh));
                atcg::Mesh::Point e1 = (p1 - pi);
                atcg::Mesh::Point e2 = (p2 - pi);
                atcg::Mesh::Point e3 = (p2 - p1);
                OpenMesh::Vec3d X_face = -face_grad_u[fh.idx()] / face_grad_u[fh.idx()].norm();

                double angle1 = acos((-e3.normalized()) | (-e2.normalized()));
                double angle2 = acos(e3.normalized() | (-e1.normalized()));

                div += (1.0 / tan(angle1)) * (e1 | X_face) + (1.0 / tan(angle2)) * (e2 | X_face);
            }
            div /= 2.0;
        }

        std::vector<std::pair<Eigen::Index, double>> bc{std::make_pair(start_vhs[0].idx(), 0.0)};
        apply_boundary_conditions(Lc, vertex_div_u, bc);
        luSolver.compute(Lc);
        Eigen::VectorXd phi = luSolver.solve(vertex_div_u);
        for(auto vh : mesh->vertices()) 
        {
            mesh->property(distance_property, vh) = phi[vh.idx()];
        }

        mesh->release_face_normals();

    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller = std::make_shared<atcg::CameraController>(aspect_ratio);

        mesh = atcg::IO::read_mesh("res/bunny.obj");
        mesh->request_vertex_colors();

        GeodesicDistanceProperty distance_property;
        mesh->add_property(distance_property);

        std::vector<VertexHandle> start_vhs;
        start_vhs.push_back(mesh->vertex_handle(0));
        double t = 0.1;

        compute_heat_geodesics(mesh, start_vhs, distance_property, t);

        cosine_colorize_mesh(mesh, distance_property, 32);

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

        atcg::EventDispatcher dispatcher(event);
    }

private:
    std::shared_ptr<atcg::CameraController> camera_controller;
    std::shared_ptr<atcg::Mesh> mesh;

    bool show_render_settings = false;
    bool render_faces = true;
    bool render_points = false;
    bool render_edges = false;
};

class G08 : public atcg::Application
{
    public:

    G08()
        :atcg::Application()
    {
        pushLayer(new G08Layer("Layer"));
    }

    ~G08() {}

};

atcg::Application* atcg::createApplication()
{
    return new G08;
}