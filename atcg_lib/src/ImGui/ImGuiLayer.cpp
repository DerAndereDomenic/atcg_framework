#include <ImGui/ImGuiLayer.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <implot.h>

#include <GLFW/glfw3.h>
#include <Core/Application.h>

#include <Renderer/Renderer.h>
#include <Renderer/Pathtracer.h>

namespace atcg
{
ImGuiLayer::ImGuiLayer() : Layer("ImGuiLayer") {}

ImGuiLayer::~ImGuiLayer() {}

void ImGuiLayer::onAttach()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();

    ImGuiStyle& style = ImGui::GetStyle();
    if(io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding              = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    GLFWwindow* window = (GLFWwindow*)Application::get()->getWindow()->getNativeWindow();

    float xscale;
    glfwGetWindowContentScale(window, &xscale, NULL);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    io.FontGlobalScale = xscale;
    style.ScaleAllSizes(xscale);
    // Hazels color theme for now (https://github.com/TheCherno/Hazel)
    auto& colors              = ImGui::GetStyle().Colors;
    colors[ImGuiCol_WindowBg] = ImVec4 {0.1f, 0.105f, 0.11f, 1.0f};

    // Headers
    colors[ImGuiCol_Header]        = ImVec4 {0.2f, 0.205f, 0.21f, 1.0f};
    colors[ImGuiCol_HeaderHovered] = ImVec4 {0.3f, 0.305f, 0.31f, 1.0f};
    colors[ImGuiCol_HeaderActive]  = ImVec4 {0.15f, 0.1505f, 0.151f, 1.0f};

    // Buttons
    colors[ImGuiCol_Button]        = ImVec4 {0.2f, 0.205f, 0.21f, 1.0f};
    colors[ImGuiCol_ButtonHovered] = ImVec4 {0.3f, 0.305f, 0.31f, 1.0f};
    colors[ImGuiCol_ButtonActive]  = ImVec4 {0.15f, 0.1505f, 0.151f, 1.0f};

    // Frame BG
    colors[ImGuiCol_FrameBg]        = ImVec4 {0.2f, 0.205f, 0.21f, 1.0f};
    colors[ImGuiCol_FrameBgHovered] = ImVec4 {0.3f, 0.305f, 0.31f, 1.0f};
    colors[ImGuiCol_FrameBgActive]  = ImVec4 {0.15f, 0.1505f, 0.151f, 1.0f};

    // Tabs
    colors[ImGuiCol_Tab]                = ImVec4 {0.15f, 0.1505f, 0.151f, 1.0f};
    colors[ImGuiCol_TabHovered]         = ImVec4 {0.38f, 0.3805f, 0.381f, 1.0f};
    colors[ImGuiCol_TabActive]          = ImVec4 {0.28f, 0.2805f, 0.281f, 1.0f};
    colors[ImGuiCol_TabUnfocused]       = ImVec4 {0.15f, 0.1505f, 0.151f, 1.0f};
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4 {0.2f, 0.205f, 0.21f, 1.0f};

    // Title
    colors[ImGuiCol_TitleBg]          = ImVec4 {0.15f, 0.1505f, 0.151f, 1.0f};
    colors[ImGuiCol_TitleBgActive]    = ImVec4 {0.15f, 0.1505f, 0.151f, 1.0f};
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4 {0.15f, 0.1505f, 0.151f, 1.0f};

    // Slider
    colors[ImGuiCol_SliderGrab]       = ImVec4 {0.2f, 0.205f, 0.21f, 1.0f};
    colors[ImGuiCol_SliderGrabActive] = ImVec4 {0.38f, 0.3805f, 0.381f, 1.0f};

    // Resize
    colors[ImGuiCol_ResizeGrip]        = ImVec4 {0.2f, 0.205f, 0.21f, 1.0f};
    colors[ImGuiCol_ResizeGripHovered] = ImVec4 {0.3f, 0.305f, 0.31f, 1.0f};
    colors[ImGuiCol_ResizeGripActive]  = ImVec4 {0.15f, 0.1505f, 0.151f, 1.0f};
}

void ImGuiLayer::onDetach()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
}

void ImGuiLayer::onEvent(Event* event)
{
    if(_block_events)
    {
        ImGuiIO& io = ImGui::GetIO();
        event->handled |= event->isInCategory(EventCategoryMouse) & io.WantCaptureMouse;
        event->handled |= event->isInCategory(EventCategoryKeyboard) & io.WantCaptureKeyboard;
    }
}

void ImGuiLayer::onImGuiRender() {}

void ImGuiLayer::begin()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if(_enable_dock_space)
    {
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
        window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                        ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowSize(viewport->Size);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::Begin("DockSpace", &_enable_dock_space, window_flags);
        ImGui::PopStyleVar(3);

        ImGuiID dockspace_id = ImGui::GetID("atcg_MainDockSpace");
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f));

        atcg::ref_ptr<atcg::Texture2D> pt_texture = Pathtracer::getOutputTexture();

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2 {0, 0});
        ImGui::Begin("Pathtracer");
        ImGui::PopStyleVar();

        const atcg::Application* app = atcg::Application::get();
        ImGui::Image((void*)(uint64_t)pt_texture->getID(), ImVec2(app->getViewportSize().x, app->getViewportSize().y));

        ImGui::End();


        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2 {0, 0});
        ImGui::Begin("Viewport");
        ImGui::PopStyleVar();

        glm::vec2 window_pos   = Application::get()->getWindow()->getPosition();
        auto viewportMinRegion = ImGui::GetWindowContentRegionMin();
        auto viewportOffset    = ImGui::GetWindowPos();

        _viewport_position = glm::ivec2(viewportMinRegion.x + viewportOffset.x - window_pos.x,
                                        viewportMinRegion.y + viewportOffset.y - window_pos.y);


        ImVec2 viewportPanelSize = ImGui::GetContentRegionAvail();
        _viewport_size           = glm::ivec2(viewportPanelSize.x, viewportPanelSize.y);

        uint64_t textureID = Renderer::getFramebuffer()->getColorAttachement(0)->getID();
        ImGui::Image(reinterpret_cast<void*>(textureID), viewportPanelSize, ImVec2 {0, 1}, ImVec2 {1, 0});

        if(ImGui::IsMouseDown(ImGuiMouseButton_Right) && ImGui::IsWindowHovered())
        {
            ImGui::SetWindowFocus("Viewport");
        }

        bool in_focus = ImGui::IsWindowFocused();

        _block_events = !(in_focus);

        ImGui::End();
    }
}

void ImGuiLayer::end()
{
    if(_enable_dock_space) ImGui::End();    // Dockspace
    ImGuiIO& io    = ImGui::GetIO();
    auto app       = Application::get();
    io.DisplaySize = ImVec2((float)app->getWindow()->getWidth(), (float)app->getWindow()->getHeight());

    // Rendering
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    if(io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        GLFWwindow* backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }
}
}    // namespace atcg