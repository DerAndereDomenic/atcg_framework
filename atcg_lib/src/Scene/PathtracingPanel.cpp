#include <Scene/PathtracingPanel.h>

#include <Core/Application.h>
#include <Pathtracing/Pathtracer.h>
#include <Pathtracing/RaytracingShaderManager.h>
#include <Core/Input.h>

#include <imgui.h>

namespace atcg
{
PathtracingPanel::PathtracingPanel(const atcg::ref_ptr<Scene>& scene) : _scene(scene) {}

void PathtracingPanel::renderPanel(const atcg::ref_ptr<PerspectiveCamera>& camera)
{
    ImGui::Begin("PT Panel");

    if(!atcg::Pathtracer::isRunning())
    {
        ImGui::DragInt("Number Samples", &_num_samples, 1.0f, 0, INT_MAX);
        ImGui::Checkbox("Viewport size", &_use_viewport_size);

        auto imgui_layer = atcg::Application::get()->getImGuiLayer();
        if(!_use_viewport_size)
        {
            ImGui::DragInt("Width", &_width, 1.0f, 1, 4096);
            ImGui::DragInt("Height", &_height, 1.0f, 1, 4096);
        }
        else
        {
            auto size = imgui_layer->getViewportSize();
            _width    = size.x;
            _height   = size.y;
        }

        if(ImGui::Button("Start") || Input::isKeyPressed(80 /*P*/))
        {
            imgui_layer->setPathtracingFocus();
            atcg::Pathtracer::draw(_scene,
                                   camera,
                                   atcg::RaytracingShaderManager::getShader("Pathtracing"),
                                   _width,
                                   _height,
                                   _num_samples);
        }

        ImGui::Separator();

        float time_seconds = Pathtracer::getLastRenderingTime();
        float time_minutes = time_seconds / 60.0f;
        float time_hours   = time_minutes / 60.0f;

        std::stringstream ss;
        ss << "Rendering took: ";

        if(time_hours >= 1.0f)
        {
            ss << (int)time_hours << "h:";
            time_minutes -= (int)time_hours * 60.0f;
            time_seconds -= (int)time_seconds * 60.0f * 60.0f;
        }

        if(time_minutes >= 1.0f)
        {
            ss << (int)time_minutes << "m:";
            time_seconds -= (int)time_minutes * 60.0f;
        }

        ss << time_seconds << "s";

        ImGui::Text(ss.str().c_str());
    }
    else
    {
        float progress = (float)atcg::Pathtracer::getFrameIndex() / (float)_num_samples;
        ImGui::ProgressBar(progress);
        // ImGui::SameLine();
        ImGui::Text((std::to_string(atcg::Pathtracer::getFrameIndex()) + "/" + std::to_string(_num_samples)).c_str());

        if(ImGui::Button("Stop"))
        {
            atcg::Pathtracer::stop();
        }
    }


    ImGui::End();
}
}    // namespace atcg