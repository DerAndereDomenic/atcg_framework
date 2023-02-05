#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <queue>

#include <numeric>

class G10Layer : public atcg::Layer
{
public:
    G10Layer(const std::string& name) : atcg::Layer(name) {}

    struct PrincipalCurvature
    {
        double k1, k2;
    };

    struct FundamentalFormFace
    {
        double e, f, g;
        atcg::Mesh::Point uf, vf;
    };

    struct LocalFrame
    {
        atcg::Mesh::Point x, y, z;
    };

    LocalFrame compute_local_frame(const atcg::Mesh::Point& localZ)
    {
        double x  = localZ[0];
        double y  = localZ[1];
        double z  = localZ[2];
        double sz = (z >= 0) ? 1 : -1;
        double a  = 1 / (sz + z);
        double ya = y * a;
        double b  = x * ya;
        double c  = x * sz;

        atcg::Mesh::Point localX = atcg::Mesh::Point {c * x * a - 1, sz * b, c};
        atcg::Mesh::Point localY = atcg::Mesh::Point {b, y * ya - sz, y};

        return {localX, localY, localZ};
    }

    // Rotate source onto target
    LocalFrame rotateCoordinateSystem(const LocalFrame& target, const LocalFrame& source)
    {
        double cosa = target.z.dot(source.z);

        double sina         = std::sqrt(std::max(0.0, 1.0 - cosa * cosa));
        atcg::Mesh::Point n = target.z.cross(source.z).normalized();
        double n1           = n[0];
        double n2           = n[1];
        double n3           = n[2];

        // https://de.wikipedia.org/wiki/Drehmatrix
        atcg::Mesh::Point r1 {n1 * n1 * (1.0f - cosa) + cosa,
                              n1 * n2 * (1.0f - cosa) - n3 * sina,
                              n1 * n3 * (1.0f - cosa) + n2 * sina};
        atcg::Mesh::Point r2 {n2 * n1 * (1.0f - cosa) + n3 * sina,
                              n2 * n2 * (1.0f - cosa) + cosa,
                              n2 * n3 * (1.0f - cosa) - n1 * sina};
        atcg::Mesh::Point r3 {n3 * n1 * (1.0f - cosa) - n2 * sina,
                              n3 * n2 * (1.0f - cosa) + n1 * sina,
                              n3 * n3 * (1.0f - cosa) + cosa};

        atcg::Mesh::Point x_new = source.x[0] * r1 + source.x[1] * r2 + source.x[2] * r3;
        atcg::Mesh::Point y_new = source.y[0] * r1 + source.y[1] * r2 + source.y[2] * r3;

        return {x_new, y_new, target.z};
    }

    void computeCurvature(const std::shared_ptr<atcg::Mesh>& mesh,
                          const OpenMesh::VPropHandleT<PrincipalCurvature>& curvature)
    {
        auto form_property = OpenMesh::makeTemporaryProperty<atcg::Mesh::FaceHandle, FundamentalFormFace>(*mesh.get());

        for(auto f_it = mesh->faces_begin(); f_it != mesh->faces_end(); ++f_it)
        {
            // Calculate properties of triangle (normal differences, edges)
            std::vector<atcg::Mesh::Point> points;
            std::vector<atcg::Mesh::Normal> normals;
            std::vector<atcg::Mesh::Point> edges(3);
            for(auto v_it: f_it->vertices())
            {
                points.push_back(mesh->point(v_it));
                normals.push_back(mesh->normal(v_it));
            }

            edges[0] = points[2] - points[1];
            edges[1] = points[0] - points[2];
            edges[2] = points[1] - points[0];

            atcg::Mesh::Point nd21  = normals[2] - normals[1];
            atcg::Mesh::Point nd02  = normals[0] - normals[2];
            atcg::Mesh::Point nd10  = normals[1] - normals[0];
            atcg::Mesh::Point nd[3] = {nd21, nd02, nd10};

            // Get orthonormal parameterization u,v (choose one edge and normalize the other)
            atcg::Mesh::Point u = edges[0].normalized();
            atcg::Mesh::Point v = (edges[1] - u.dot(edges[1]) * u).normalized();

            // Construct least squares matrix
            Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, 3);
            Eigen::VectorXd b = Eigen::VectorXd::Zero(6);

            for(uint32_t i = 0; i < 3; ++i)    // One equation for each edge
            {
                double a1 = edges[i].dot(u);
                double a2 = edges[i].dot(v);
                double b1 = nd[i].dot(u);
                double b2 = nd[i].dot(v);

                A(2 * i, 0)     = a1;
                A(2 * i, 1)     = a2;
                A(2 * i + 1, 1) = a1;
                A(2 * i + 1, 2) = a2;

                b(2 * i)     = b1;
                b(2 * i + 1) = b2;
            }

            Eigen::VectorXd form = A.colPivHouseholderQr().solve(b);

            FundamentalFormFace form_data;
            form_data.e          = form(0);
            form_data.f          = form(1);
            form_data.g          = form(2);
            form_data.uf         = u;
            form_data.vf         = v;
            form_property[*f_it] = form_data;
        }

        //"Average" tensors over vertices
        for(auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end(); ++v_it)
        {
            atcg::Mesh::Normal v_normal = mesh->normal(*v_it);
            LocalFrame local_frame_p    = compute_local_frame(v_normal);

            double ep = 0, fp = 0, gp = 0;
            uint32_t num_faces = 0;
            for(auto f_it = v_it->faces().begin(); f_it != v_it->faces().end(); ++f_it)
            {
                FundamentalFormFace form = form_property[*f_it];
                LocalFrame local_frame_f = {form.uf, form.vf, form.uf.cross(form.vf).normalized()};

                LocalFrame rotated_frame = rotateCoordinateSystem(local_frame_p, local_frame_f);

                Eigen::Matrix2d F;
                F(0, 0) = form.e;
                F(0, 1) = form.f;
                F(1, 0) = form.f;
                F(1, 1) = form.g;
                Eigen::Vector2d up, vp;
                up(0) = local_frame_p.x.dot(rotated_frame.x);
                up(1) = local_frame_p.x.dot(rotated_frame.y);
                vp(0) = local_frame_p.y.dot(rotated_frame.x);
                vp(1) = local_frame_p.y.dot(rotated_frame.y);

                ep += up.transpose() * F * up;
                fp += up.transpose() * F * vp;
                gp += vp.transpose() * F * vp;
                ++num_faces;
            }

            ep /= static_cast<double>(num_faces);
            fp /= static_cast<double>(num_faces);
            gp /= static_cast<double>(num_faces);

            double ev1 = (ep + gp) / 2.0f + std::sqrt(((ep + gp) / 2.0f) * ((ep + gp) / 2.0f) - ep * gp + fp * fp);
            double ev2 = (ep + gp) / 2.0f - std::sqrt(((ep + gp) / 2.0f) * ((ep + gp) / 2.0f) - ep * gp + fp * fp);
            // double mean_curvature = (ev1 + ev2) / 2.0f;
            mesh->property(curvature, *v_it) = {ev1, ev2};
        }
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = std::make_shared<atcg::CameraController>(aspect_ratio);

        mesh = atcg::IO::read_mesh("res/bunny.obj");
        mesh->request_vertex_normals();
        mesh->request_face_normals();
        mesh->update_normals();
        mesh->request_vertex_colors();

        OpenMesh::VPropHandleT<PrincipalCurvature> property_curvature;
        mesh->add_property(property_curvature);

        computeCurvature(mesh, property_curvature);

        double min_curvature = std::numeric_limits<double>::infinity();
        double max_curvature = -std::numeric_limits<double>::infinity();

        for(auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end(); ++v_it)
        {
            double curvature =
                (mesh->property(property_curvature, *v_it).k1 + mesh->property(property_curvature, *v_it).k2) / 2.0f;
            min_curvature = std::min(min_curvature, curvature);
            max_curvature = std::max(max_curvature, curvature);
        }

        std::cout << max_curvature << "\n";
        std::cout << min_curvature << "\n";

        double max_abs_value = std::max(max_curvature, -min_curvature);
        for(auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end(); ++v_it)
        {
            double curvature =
                (mesh->property(property_curvature, *v_it).k1 + mesh->property(property_curvature, *v_it).k2) / 2.0f;
            if(curvature > 0)
                mesh->set_color(*v_it, {curvature / max_abs_value * 255, 0, 0});
            else
                mesh->set_color(*v_it, {0, 0, -curvature / max_abs_value * 255});
        }

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

        if(mesh && render_edges) atcg::Renderer::drawLines(mesh, glm::vec3(1), camera_controller->getCamera());
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
    virtual void onEvent(atcg::Event& event) override { camera_controller->onEvent(event); }

private:
    std::shared_ptr<atcg::CameraController> camera_controller;
    std::shared_ptr<atcg::Mesh> mesh;

    bool show_render_settings = true;
    bool render_faces         = true;
    bool render_points        = false;
    bool render_edges         = false;
};

class G10 : public atcg::Application
{
public:
    G10() : atcg::Application() { pushLayer(new G10Layer("Layer")); }

    ~G10() {}
};

atcg::Application* atcg::createApplication()
{
    return new G10;
}