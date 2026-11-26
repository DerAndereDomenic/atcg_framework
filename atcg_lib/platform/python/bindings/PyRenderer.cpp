#include "Bindings.h"

#include <Renderer/CompileData.h>
#include <Renderer/RenderPassRegistry.h>

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
// Pipeline states, graphics commands, renderer, shaders, textures, buffers and render graphs
void bindRenderer(BindingContext& ctx)
{
    py::module_ m = ctx.module();

    auto m_renderer = m.def_submodule("Renderer");

    auto m_graphics_api    = m.def_submodule("GraphicsCommand");
    auto m_renderer_system = py::class_<atcg::RendererSystem, atcg::ref_ptr<atcg::RendererSystem>>(m, "RendererSystem");
    auto m_context         = py::class_<atcg::Context, atcg::ref_ptr<atcg::Context>>(m, "Context");
    auto m_shader_type     = py::enum_<atcg::ShaderType>(m, "ShaderType");
    auto m_shader          = py::class_<atcg::Shader, atcg::Asset, atcg::ref_ptr<atcg::Shader>>(m, "Shader");
    auto m_shader_manager  = m.def_submodule("ShaderManager");
    auto m_shader_manager_system =
        py::class_<atcg::ShaderManagerSystem, atcg::ref_ptr<atcg::ShaderManagerSystem>>(m, "ShaderManagerSystem");
    auto m_texture_format        = py::enum_<atcg::TextureFormat>(m, "TextureFormat");
    auto m_texture_wrap_mode     = py::enum_<atcg::TextureWrapMode>(m, "TextureWrapMode");
    auto m_texture_filter_mode   = py::enum_<atcg::TextureFilterMode>(m, "TextureFilterMode");
    auto m_texture_sampler       = py::class_<atcg::TextureSamplerSpecification>(m, "TextureSampler");
    auto m_texture_specification = py::class_<atcg::TextureSpecification>(m, "TextureSpecification");
    auto m_image        = py::class_<atcg::Image, atcg::ref_ptr<atcg::Image>>(m, "Image", py::buffer_protocol());
    auto m_texture_type = py::enum_<atcg::TextureType>(m, "TextureType");
    auto m_texture      = py::class_<atcg::Texture, atcg::ref_ptr<atcg::Texture>>(m, "Texture");
    auto m_texture2d =
        py::class_<atcg::Texture2D, atcg::Texture, atcg::Asset, atcg::ref_ptr<atcg::Texture2D>>(m, "Texture2D");
    auto m_texture3d =
        py::class_<atcg::Texture3D, atcg::Texture, atcg::Asset, atcg::ref_ptr<atcg::Texture3D>>(m, "Texture3D");
    auto m_texture_cube =
        py::class_<atcg::TextureCube, atcg::Texture, atcg::ref_ptr<atcg::TextureCube>>(m, "TextureCube");
    auto m_texture_array =
        py::class_<atcg::TextureArray, atcg::Texture, atcg::ref_ptr<atcg::TextureArray>>(m, "TextureArray");
    auto m_texture_cube_array =
        py::class_<atcg::TextureCubeArray, atcg::Texture, atcg::ref_ptr<atcg::TextureCubeArray>>(m, "TextureCubeArray");
    auto m_texture2d_multisample =
        py::class_<atcg::Texture2DMultiSample, atcg::Texture, atcg::ref_ptr<atcg::Texture2DMultiSample>>(m,
                                                                                                         "Texture2DMult"
                                                                                                         "iSample");
    auto m_framebuffer_texture_specification =
        py::class_<atcg::FramebufferTextureSpecification>(m, "FramebufferTextureSpecification");
    auto m_framebuffer_specification = py::class_<atcg::FramebufferSpecification>(m, "FramebufferSpecification");
    auto m_framebuffer = py::class_<atcg::Framebuffer, atcg::ref_ptr<atcg::Framebuffer>>(m, "Framebuffer");
    auto m_skybox      = py::class_<atcg::Skybox, atcg::ref_ptr<atcg::Skybox>>(m, "Skybox");

    auto m_vertex_buffer = py::class_<atcg::VertexBuffer, atcg::ref_ptr<atcg::VertexBuffer>>(m, "VertexBuffer");
    auto m_index_buffer =
        py::class_<atcg::IndexBuffer, atcg::VertexBuffer, atcg::ref_ptr<atcg::IndexBuffer>>(m, "IndexBuffer");
    auto m_vertex_array     = py::class_<atcg::VertexArray, atcg::ref_ptr<atcg::VertexArray>>(m, "VertexArray");
    auto m_buffer_layout    = py::class_<atcg::BufferLayout>(m, "BufferLayout");
    auto m_buffer_element   = py::class_<atcg::BufferElement>(m, "BufferElement");
    auto m_shader_data_type = py::enum_<atcg::ShaderDataType>(m, "ShaderDataType");

    auto m_cull_mode          = py::enum_<atcg::CullMode>(m, "CullMode");
    auto m_primitive_topology = py::enum_<atcg::PrimitiveTopology>(m, "PrimitiveTopology");
    auto m_depth_function     = py::enum_<atcg::DepthFunction>(m, "DepthFunction");
    auto m_depth_state        = py::class_<atcg::DepthState>(m, "DepthState");
    auto m_blend_state        = py::class_<atcg::BlendState>(m, "BlendState");
    auto m_rasterizer_state   = py::class_<atcg::RasterizerState>(m, "RasterizerState");
    auto m_graphics_pipeline  = py::class_<atcg::GraphicsPipeline>(m, "GraphicsPipeline");

    auto m_compile_data         = py::class_<atcg::CompileData>(m, "CompileData");
    auto m_render_pass          = py::class_<atcg::RenderPass, atcg::ref_ptr<atcg::RenderPass>>(m, "RenderPass");
    auto m_render_graph         = py::class_<atcg::RenderGraph, atcg::ref_ptr<atcg::RenderGraph>>(m, "RenderGraph");
    auto m_render_pass_registry = m.def_submodule("RenderPassRegistry");

    ctx.define(
        [=]() mutable
        {
            m_cull_mode.value("ATCG_FRONT_FACE_CULLING", atcg::CullMode::ATCG_FRONT_FACE_CULLING)
                .value("ATCG_BACK_FACE_CULLING", atcg::CullMode::ATCG_BACK_FACE_CULLING)
                .value("ATCG_BOTH_FACE_CULLING", atcg::CullMode::ATCG_BOTH_FACE_CULLING)
                .value("ATCG_NO_CULLING", atcg::CullMode::ATCG_NO_CULLING)
                .export_values();

            m_primitive_topology.value("ATCG_POINTS", atcg::PrimitiveTopology::ATCG_POINTS)
                .value("ATCG_TRIANGLES", atcg::PrimitiveTopology::ATCG_TRIANGLES)
                .value("ATCG_LINES", atcg::PrimitiveTopology::ATCG_LINES)
                .export_values();

            m_depth_function.value("ATCG_LESS", atcg::DepthFunction::ATCG_LESS)
                .value("ATCG_LEQUAL", atcg::DepthFunction::ATCG_LEQUAL)
                .value("ATCG_EQUAL", atcg::DepthFunction::ATCG_EQUAL)
                .value("ATCG_GREATER", atcg::DepthFunction::ATCG_GREATER)
                .value("ATCG_GEQUAL", atcg::DepthFunction::ATCG_GEQUAL)
                .value("ATCG_ALWAYS", atcg::DepthFunction::ATCG_ALWAYS)
                .value("ATCG_NEVER", atcg::DepthFunction::ATCG_NEVER)
                .value("ATCG_NOTEQUAL", atcg::DepthFunction::ATCG_NOTEQUAL)
                .export_values();

            m_depth_state.def(py::init<>())
                .def_readwrite("depth_testing_enabled", &atcg::DepthState::depth_testing_enabled)
                .def_readwrite("depth_write_enabled", &atcg::DepthState::depth_write_enabled)
                .def_readwrite("depth_function", &atcg::DepthState::depth_function)
                .def("setDepthFunction", &atcg::DepthState::setDepthFunction, "function"_a)
                .def("enableDepthTesting", &atcg::DepthState::enableDepthTesting, "enable"_a = true)
                .def("enableDepthWrite", &atcg::DepthState::enableDepthWrite, "enable"_a = true)
                .def(
                    "__eq__",
                    [](const atcg::DepthState& self, const atcg::DepthState& other) { return self == other; },
                    py::is_operator());

            m_blend_state.def(py::init<>())
                .def_readwrite("blend_enabled", &atcg::BlendState::blend_enabled)
                .def("enableBlending", &atcg::BlendState::enableBlending, "enable"_a = true)
                .def(
                    "__eq__",
                    [](const atcg::BlendState& self, const atcg::BlendState& other) { return self == other; },
                    py::is_operator());

            m_rasterizer_state.def(py::init<>())
                .def_readwrite("cull_mode", &atcg::RasterizerState::cull_mode)
                .def_readwrite("culling_enabled", &atcg::RasterizerState::culling_enabled)
                .def_readwrite("depth_state", &atcg::RasterizerState::depth_state)
                .def_readwrite("blend_state", &atcg::RasterizerState::blend_state)
                .def_readwrite("point_size", &atcg::RasterizerState::point_size)
                .def_readwrite("line_size", &atcg::RasterizerState::line_size)
                .def("setCullMode", &atcg::RasterizerState::setCullMode, "mode"_a)
                .def("enableCulling", &atcg::RasterizerState::enableCulling, "enable"_a = true)
                .def("setPointSize", &atcg::RasterizerState::setPointSize, "size"_a)
                .def("setLineSize", &atcg::RasterizerState::setLineSize, "size"_a)
                .def("setDepthState", &atcg::RasterizerState::setDepthState, "depth_state"_a)
                .def("setBlendState", &atcg::RasterizerState::setBlendState, "blend_state"_a)
                .def(
                    "__eq__",
                    [](const atcg::RasterizerState& self, const atcg::RasterizerState& other) { return self == other; },
                    py::is_operator());

            m_graphics_pipeline.def(py::init<>())
                .def_readwrite("shader", &atcg::GraphicsPipeline::shader)
                .def_readwrite("rasterizer_state", &atcg::GraphicsPipeline::rasterizer_state)
                .def_readwrite("primitive_topology", &atcg::GraphicsPipeline::primitive_topology)
                .def("setShader", &atcg::GraphicsPipeline::setShader, "shader"_a)
                .def("setRasterizerState", &atcg::GraphicsPipeline::setRasterizerState, "rasterizer_state"_a)
                .def("setPrimitiveTopology", &atcg::GraphicsPipeline::setPrimitiveTopology, "topology"_a)
                .def(
                    "__eq__",
                    [](const atcg::GraphicsPipeline& self, const atcg::GraphicsPipeline& other)
                    { return self == other; },
                    py::is_operator());

            m_graphics_api.def("beginRenderPass", &atcg::GraphicsCommand::beginRenderPass, "target_framebuffer"_a)
                .def("endRenderPass", &atcg::GraphicsCommand::endRenderPass)
                .def("bindPipeline", &atcg::GraphicsCommand::bindPipeline, "pipeline"_a)
                .def("setViewport", &atcg::GraphicsCommand::setViewport, "x"_a, "y"_a, "width"_a, "height"_a)
                .def("getViewport", &atcg::GraphicsCommand::getViewport)
                .def("bindVertexArray", &atcg::GraphicsCommand::bindVertexArray, "vao"_a)
                .def("bindTexture", &atcg::GraphicsCommand::bindTexture, "slot"_a, "texture"_a)
                .def("bindStorageBuffer", &atcg::GraphicsCommand::bindStorageBuffer, "slot"_a, "buffer"_a)
                .def("draw", &atcg::GraphicsCommand::draw, "vertex_count"_a)
                .def("drawIndexed", &atcg::GraphicsCommand::drawIndexed, "index_count"_a)
                .def("drawInstanced", &atcg::GraphicsCommand::drawInstanced, "vertex_count"_a, "num_instances"_a)
                .def("drawIndexedInstanced",
                     &atcg::GraphicsCommand::drawIndexedInstanced,
                     "index_count"_a,
                     "num_instances"_a)
                .def("setClearColor", &atcg::GraphicsCommand::setClearColor, "color"_a)
                .def("setClearDepth", &atcg::GraphicsCommand::setClearDepth, "depth"_a = 1.0f)
                .def("clear", &atcg::GraphicsCommand::clear)
                .def("finish", &atcg::GraphicsCommand::finish)
                .def("getTotalTextureUnits", &atcg::GraphicsCommand::getTotalTextureUnits);

            m_renderer.def("init", &atcg::Renderer::init, "width"_a, "height"_a, "context"_a, "shader_manager"_a)
                .def("use", &atcg::Renderer::use)
                .def("finishFrame", &atcg::Renderer::finishFrame)
                .def("finish", &atcg::Renderer::finish)
                .def("drawVAO",
                     &atcg::Renderer::drawVAO,
                     "vao"_a,
                     "camera"_a,
                     "model"_a,
                     "pipeline"_a,
                     "size"_a,
                     "instances"_a = 1)
                .def("drawCADGrid", &atcg::Renderer::drawCADGrid, "camera"_a, "transparency"_a = 0.6f)
                .def("drawCircle",
                     &atcg::Renderer::drawCircle,
                     "position"_a,
                     "radius"_a,
                     "thickness"_a,
                     "color"_a,
                     "camera"_a    = py::none(),
                     "entity_id"_a = static_cast<uint32_t>(-1))
                .def("drawImage",
                     py::overload_cast<const atcg::ref_ptr<atcg::Framebuffer>&>(&atcg::Renderer::drawImage),
                     "img"_a)
                .def("drawImage",
                     py::overload_cast<const atcg::ref_ptr<atcg::Texture2D>&, const atcg::ref_ptr<atcg::Texture2D>&>(
                         &atcg::Renderer::drawImage),
                     "img"_a,
                     "entity_ids"_a = py::none())
                .def("getFramebuffer", &atcg::Renderer::getFramebuffer)
                .def("getShaderManager", &atcg::Renderer::getShaderManager)
                .def("resize", &atcg::Renderer::resize, "width"_a, "height"_a)
                .def("getFrameCounter", &atcg::Renderer::getFrameCounter)
                .def("popTextureID", &atcg::Renderer::popTextureID)
                .def("pushTextureID", &atcg::Renderer::pushTextureID, "id"_a);

            m_renderer_system.def(py::init<>())
                .def("init", &atcg::RendererSystem::init, "width"_a, "height"_a, "context"_a, "shader_manager"_a)
                .def("use", &atcg::RendererSystem::use)
                .def("finishFrame", &atcg::RendererSystem::finishFrame)
                .def("finish", &atcg::RendererSystem::finish)
                .def("drawVAO",
                     &atcg::RendererSystem::drawVAO,
                     "vao"_a,
                     "camera"_a,
                     "model"_a,
                     "pipeline"_a,
                     "size"_a,
                     "instances"_a = 1)
                .def("drawCADGrid", &atcg::RendererSystem::drawCADGrid, "camera"_a, "transparency"_a = 0.6f)
                .def("drawCircle",
                     &atcg::RendererSystem::drawCircle,
                     "position"_a,
                     "radius"_a,
                     "thickness"_a,
                     "color"_a,
                     "camera"_a    = py::none(),
                     "entity_id"_a = static_cast<uint32_t>(-1))
                .def("drawImage",
                     py::overload_cast<const atcg::ref_ptr<atcg::Framebuffer>&>(&atcg::RendererSystem::drawImage),
                     "img"_a)
                .def("drawImage",
                     py::overload_cast<const atcg::ref_ptr<atcg::Texture2D>&, const atcg::ref_ptr<atcg::Texture2D>&>(
                         &atcg::RendererSystem::drawImage),
                     "img"_a,
                     "entity_ids"_a = py::none())
                .def("getFramebuffer", &atcg::RendererSystem::getFramebuffer)
                .def("getShaderManager", &atcg::RendererSystem::getShaderManager)
                .def("resize", &atcg::RendererSystem::resize, "width"_a, "height"_a)
                .def("getFrameCounter", &atcg::RendererSystem::getFrameCounter)
                .def("popTextureID", &atcg::RendererSystem::popTextureID)
                .def("pushTextureID", &atcg::RendererSystem::pushTextureID, "id"_a);

            m_shader_type.value("None", atcg::ShaderType::None)
                .value("VERTEX", atcg::ShaderType::VERTEX)
                .value("FRAGMENT", atcg::ShaderType::FRAGMENT)
                .value("GEOMETRY", atcg::ShaderType::GEOMETRY)
                .value("COMPUTE", atcg::ShaderType::COMPUTE);

            m_shader.def(py::init<>())
                .def(py::init<std::string>(), "compute_path"_a)
                .def(py::init<std::string, std::string>(), "vertex_path"_a, "fragment_path"_a)
                .def(py::init<std::string, std::string, std::string>(),
                     "vertex_path"_a,
                     "fragment_path"_a,
                     "geometry_path"_a)
                .def_static("getStaticType", &atcg::Shader::getStaticType)
                .def(
                    "recompile",
                    [](const atcg::ref_ptr<atcg::Shader>& self, const std::string& compute_path)
                    { self->recompile(compute_path); },
                    "compute_path"_a)
                .def(
                    "recompile",
                    [](const atcg::ref_ptr<atcg::Shader>& self,
                       const std::string& vertex_path,
                       const std::string& fragment_path) { self->recompile(vertex_path, fragment_path); },
                    "vertex_path"_a,
                    "fragment_path"_a)
                .def(
                    "recompile",
                    [](const atcg::ref_ptr<atcg::Shader>& self,
                       const std::string& vertex_path,
                       const std::string& fragment_path,
                       const std::string& geometry_path)
                    { self->recompile(vertex_path, fragment_path, geometry_path); },
                    "vertex_path"_a,
                    "fragment_path"_a,
                    "geometry_path"_a)
                .def("setInt", &atcg::Shader::setInt, "uniform_name"_a, "value"_a)
                .def("setFloat", &atcg::Shader::setFloat, "uniform_name"_a, "value"_a)
                .def("setVec2", &atcg::Shader::setVec2, "uniform_name"_a, "value"_a)
                .def("setVec3", &atcg::Shader::setVec3, "uniform_name"_a, "value"_a)
                .def("setVec4", &atcg::Shader::setVec4, "uniform_name"_a, "value"_a)
                .def("setMat4", &atcg::Shader::setMat4, "uniform_name"_a, "value"_a)
                .def("selectSubroutine", &atcg::Shader::selectSubroutine, "subroutine_type"_a, "subroutine_name"_a)
                .def("setMVP",
                     &atcg::Shader::setMVP,
                     "model"_a      = glm::mat4(1),
                     "view"_a       = glm::mat4(1),
                     "projection"_a = glm::mat4(1))
                .def("dispatch", &atcg::Shader::dispatch, "work_groups"_a)
                .def("getSource", &atcg::Shader::getSource, "type"_a)
                .def("hasGeometryShader", &atcg::Shader::hasGeometryShader)
                .def("isComputeShader", &atcg::Shader::isComputeShader)
                .def("getVertexPath", &atcg::Shader::getVertexPath)
                .def("getGeometryPath", &atcg::Shader::getGeometryPath)
                .def("getFragmentPath", &atcg::Shader::getFragmentPath)
                .def("getComputePath", &atcg::Shader::getComputePath);

            m_shader_manager.def("getShader", &atcg::ShaderManager::getShader, "name"_a)
                .def("addShader", &atcg::ShaderManager::addShader, "name"_a, "shader"_a)
                .def("addShaderFromName", &atcg::ShaderManager::addShaderFromName, "name"_a)
                .def("addComputeShaderFromName", &atcg::ShaderManager::addComputeShaderFromName, "name"_a)
                .def("hasShader", &atcg::ShaderManager::hasShader, "name"_a)
                .def("onUpdate", &atcg::ShaderManager::onUpdate)
                .def(
                    "setShaderPath",
                    [](const std::string& path)
                    { atcg::SystemRegistry::instance()->getSystem<atcg::ShaderManagerSystem>()->setShaderPath(path); },
                    "path"_a)
                .def("getShaderPath",
                     []() {
                         return atcg::SystemRegistry::instance()
                             ->getSystem<atcg::ShaderManagerSystem>()
                             ->getShaderPath()
                             .string();
                     });

            m_shader_manager_system.def("getShader", &atcg::ShaderManagerSystem::getShader, "name"_a)
                .def("addShader", &atcg::ShaderManagerSystem::addShader, "name"_a, "shader"_a)
                .def("addShaderFromName", &atcg::ShaderManagerSystem::addShaderFromName, "name"_a)
                .def("addComputeShaderFromName", &atcg::ShaderManagerSystem::addComputeShaderFromName, "name"_a)
                .def("hasShader", &atcg::ShaderManagerSystem::hasShader, "name"_a)
                .def("onUpdate", &atcg::ShaderManagerSystem::onUpdate)
                .def(
                    "setShaderPath",
                    [](const atcg::ref_ptr<atcg::ShaderManagerSystem>& self, const std::string& path)
                    { self->setShaderPath(path); },
                    "path"_a)
                .def("getShaderPath",
                     [](const atcg::ref_ptr<atcg::ShaderManagerSystem>& self)
                     { return self->getShaderPath().string(); });

            m_texture_format.value("RG", atcg::TextureFormat::RG)
                .value("RGB", atcg::TextureFormat::RGB)
                .value("RGBA", atcg::TextureFormat::RGBA)
                .value("RGFLOAT", atcg::TextureFormat::RGFLOAT)
                .value("RGBFLOAT", atcg::TextureFormat::RGBFLOAT)
                .value("RGBAFLOAT", atcg::TextureFormat::RGBAFLOAT)
                .value("RINT", atcg::TextureFormat::RINT)
                .value("RINT8", atcg::TextureFormat::RINT8)
                .value("RFLOAT", atcg::TextureFormat::RFLOAT)
                .value("DEPTH", atcg::TextureFormat::DEPTH);

            m_texture_wrap_mode.value("REPEAT", atcg::TextureWrapMode::REPEAT)
                .value("CLAMP_TO_EDGE", atcg::TextureWrapMode::CLAMP_TO_EDGE)
                .value("BORDER", atcg::TextureWrapMode::BORDER);

            m_texture_filter_mode.value("NEAREST", atcg::TextureFilterMode::NEAREST)
                .value("LINEAR", atcg::TextureFilterMode::LINEAR)
                .value("MIPMAP_LINEAR", atcg::TextureFilterMode::MIPMAP_LINEAR);

            m.def("textureFormatToString", &atcg::textureFormatToString, "format"_a);
            m.def("stringToTextureFormat", &atcg::stringToTextureFormat, "str"_a);
            m.def("textureWrapModeToString", &atcg::textureWrapModeToString, "mode"_a);
            m.def("stringToTextureWrapMode", &atcg::stringToTextureWrapMode, "str"_a);
            m.def("textureFilterModeToString", &atcg::textureFilterModeToString, "mode"_a);
            m.def("stringToTextureFilterMode", &atcg::stringToTextureFilterMode, "str"_a);

            m_texture_sampler.def(py::init<>())
                .def(py::init<>(
                         [](atcg::TextureFilterMode filter_mode, atcg::TextureWrapMode wrap_mode, bool mip_map)
                         {
                             atcg::TextureSamplerSpecification sampler;
                             sampler.filter_mode = filter_mode;
                             sampler.wrap_mode   = wrap_mode;
                             sampler.mip_map     = mip_map;
                             return sampler;
                         }),
                     py::arg_v("filter_mode", atcg::TextureFilterMode::LINEAR, "TextureFilterMode.LINEAR"),
                     py::arg_v("wrap_mode", atcg::TextureWrapMode::REPEAT, "TextureWrapMode.REPEAT"),
                     "mip_map"_a = false)
                .def_readwrite("wrap_mode", &atcg::TextureSamplerSpecification::wrap_mode)
                .def_readwrite("filter_mode", &atcg::TextureSamplerSpecification::filter_mode)
                .def_readwrite("mip_map", &atcg::TextureSamplerSpecification::mip_map);

            m_texture_specification.def(py::init<>())
                .def(py::init<atcg::TextureFormat>(), "format"_a)
                .def(py::init<uint32_t, uint32_t, atcg::TextureFormat>(), "width"_a, "height"_a, "format"_a)
                .def(py::init<uint32_t, uint32_t, uint32_t, atcg::TextureFormat>(),
                     "width"_a,
                     "height"_a,
                     "depth"_a,
                     "format"_a)
                .def(py::init<uint32_t, uint32_t, uint32_t, atcg::TextureFormat, atcg::TextureSamplerSpecification>(),
                     "width"_a,
                     "height"_a,
                     "depth"_a,
                     "format"_a,
                     "sampler"_a)
                .def_readwrite("width", &atcg::TextureSpecification::width)
                .def_readwrite("height", &atcg::TextureSpecification::height)
                .def_readwrite("depth", &atcg::TextureSpecification::depth)
                .def_readwrite("num_samples", &atcg::TextureSpecification::num_samples)
                .def_readwrite("sampler", &atcg::TextureSpecification::sampler)
                .def_readwrite("format", &atcg::TextureSpecification::format)
                .def("pixelSize", &atcg::TextureSpecification::pixelSize)
                .def("channelSize", &atcg::TextureSpecification::channelSize)
                .def("numChannels", &atcg::TextureSpecification::numChannels)
                .def("isFloat", &atcg::TextureSpecification::isFloat)
                .def("isInt", &atcg::TextureSpecification::isInt);

            m_image.def(py::init<>())
                .def(py::init<const torch::Tensor&>(), "tensor"_a)
                .def("load", &atcg::Image::load, "filename"_a)
                .def("store", &atcg::Image::store, "filename"_a)
                .def("applyGamma", &atcg::Image::applyGamma, "gamma"_a)
                .def("setData", py::overload_cast<const torch::Tensor&>(&atcg::Image::setData), "data"_a)
                .def("data", &atcg::Image::data)
                .def("width", &atcg::Image::width)
                .def("height", &atcg::Image::height)
                .def("channels", &atcg::Image::channels)
                .def("name", &atcg::Image::name)
                .def("isHDR", &atcg::Image::isHDR)
                .def_buffer(
                    [](const atcg::Image& img) -> py::buffer_info
                    {
                        bool isHDR  = img.isHDR();
                        size_t size = isHDR ? sizeof(float) : sizeof(uint8_t);
                        void* data  = img.data().data_ptr();
                        return py::buffer_info(data,
                                               size,
                                               isHDR ? py::format_descriptor<float>::format()
                                                     : py::format_descriptor<uint8_t>::format(),
                                               3,
                                               {img.height(), img.width(), img.channels()},
                                               {img.width() * img.channels() * size, img.channels() * size, size});
                    });

            m.def("imread", &atcg::IO::imread, "filename"_a, "gamma"_a = 1.0f);
            m.def("imwrite", &atcg::IO::imwrite, "image"_a, "filename"_a, "gamma"_a = 1.0f);

            m_texture_type.value("TEXTURE_2D", atcg::TextureType::TEXTURE_2D)
                .value("TEXTURE_3D", atcg::TextureType::TEXTURE_3D)
                .value("TEXTURE_CUBE", atcg::TextureType::TEXTURE_CUBE)
                .value("TEXTURE_ARRAY", atcg::TextureType::TEXTURE_ARRAY)
                .value("TEXTURE_CUBE_ARRAY", atcg::TextureType::TEXTURE_CUBE_ARRAY)
                .value("TEXTURE_2D_MULTISAMPLE", atcg::TextureType::TEXTURE_2D_MULTISAMPLE);

            m_texture.def_static("create", &atcg::Texture::create, "type"_a, "specification"_a)
                .def("setData", py::overload_cast<const torch::Tensor&>(&atcg::Texture::setData), "data"_a)
                .def("setData",
                     py::overload_cast<const atcg::ref_ptr<atcg::VertexBuffer>&>(&atcg::Texture::setData),
                     "data"_a)
                .def(
                    "getData",
                    [](const atcg::ref_ptr<atcg::Texture>& self,
                       std::optional<torch::Device> device,
                       const uint32_t mip_level)
                    { return self->getData(device.value_or(torch::Device(atcg::GPU)), mip_level); },
                    "device"_a    = py::none(),
                    "mip_level"_a = 0)
                .def(
                    "__getitem__",
                    [](const atcg::ref_ptr<atcg::Texture>& self, py::object index) -> py::object
                    {
                        torch::Tensor data = self->getData(torch::Device(atcg::GPU));
                        return py::cast(data).attr("__getitem__")(index);
                    },
                    "index"_a)
                .def(
                    "__setitem__",
                    [](const atcg::ref_ptr<atcg::Texture>& self, py::object index, py::object value)
                    {
                        // Download the texture, assign the slice in place and upload the result
                        torch::Tensor data = self->getData(torch::Device(atcg::GPU));
                        py::cast(data).attr("__setitem__")(index, value);
                        self->setData(data);
                    },
                    "index"_a,
                    "value"_a)
                .def("width", &atcg::Texture::width)
                .def("height", &atcg::Texture::height)
                .def("depth", &atcg::Texture::depth)
                .def("channels", &atcg::Texture::channels)
                .def("isHDR", &atcg::Texture::isHDR)
                .def("getID", &atcg::Texture::getID)
                .def("getSpecification", &atcg::Texture::getSpecification)
                .def("useForCompute", &atcg::Texture::useForCompute, "slot"_a = 0)
                .def("generateMipmaps", &atcg::Texture::generateMipmaps)
                .def("clone", &atcg::Texture::clone)
                .def("swap", &atcg::Texture::swap, "target"_a)
                .def("isDeviceMapped", &atcg::Texture::isDeviceMapped)
                .def("unmapDevicePointers", &atcg::Texture::unmapDevicePointers)
                .def("unmapPointers", &atcg::Texture::unmapPointers);

            m_texture2d
                .def_static(
                    "create",
                    [](const atcg::TextureSpecification& spec) { return atcg::Texture2D::create(spec); },
                    "specification"_a)
                .def_static(
                    "create",
                    [](const atcg::ref_ptr<atcg::Image>& img, const atcg::TextureSpecification& spec)
                    { return atcg::Texture2D::create(img, spec); },
                    "img"_a,
                    "specification"_a)
                .def_static(
                    "create",
                    [](const atcg::ref_ptr<atcg::Image>& img) { return atcg::Texture2D::create(img); },
                    "img"_a)
                .def_static(
                    "create",
                    [](const torch::Tensor& img) { return atcg::Texture2D::create(img); },
                    "img"_a)
                .def_static(
                    "create",
                    [](const torch::Tensor& img, const atcg::TextureSpecification& spec)
                    { return atcg::Texture2D::create(img, spec); },
                    "img"_a,
                    "specification"_a)
                .def_static("getStaticType", &atcg::Texture2D::getStaticType);

            m_texture3d
                .def_static(
                    "create",
                    [](const atcg::TextureSpecification& spec) { return atcg::Texture3D::create(spec); },
                    "specification"_a)
                .def_static(
                    "create",
                    [](const torch::Tensor& img) { return atcg::Texture3D::create(img); },
                    "img"_a)
                .def_static(
                    "create",
                    [](const torch::Tensor& img, const atcg::TextureSpecification& spec)
                    { return atcg::Texture3D::create(img, spec); },
                    "img"_a,
                    "specification"_a)
                .def_static("getStaticType", &atcg::Texture3D::getStaticType);

            m_texture_cube
                .def_static(
                    "create",
                    [](const atcg::TextureSpecification& spec) { return atcg::TextureCube::create(spec); },
                    "specification"_a)
                .def_static(
                    "create",
                    [](const torch::Tensor& img) { return atcg::TextureCube::create(img); },
                    "img"_a);

            m_texture_array
                .def_static(
                    "create",
                    [](const atcg::TextureSpecification& spec) { return atcg::TextureArray::create(spec); },
                    "specification"_a)
                .def_static(
                    "create",
                    [](const torch::Tensor& img) { return atcg::TextureArray::create(img); },
                    "img"_a)
                .def_static(
                    "create",
                    [](const torch::Tensor& img, const atcg::TextureSpecification& spec)
                    { return atcg::TextureArray::create(img, spec); },
                    "img"_a,
                    "specification"_a);

            m_texture_cube_array
                .def_static(
                    "create",
                    [](const atcg::TextureSpecification& spec) { return atcg::TextureCubeArray::create(spec); },
                    "specification"_a)
                .def_static(
                    "create",
                    [](const torch::Tensor& img) { return atcg::TextureCubeArray::create(img); },
                    "img"_a);

            m_texture2d_multisample.def_static(
                "create",
                [](const atcg::TextureSpecification& spec) { return atcg::Texture2DMultiSample::create(spec); },
                "specification"_a);

            m_framebuffer_texture_specification.def(py::init<>())
                .def(py::init<atcg::TextureSpecification, atcg::TextureType>(),
                     "spec"_a,
                     "type"_a = atcg::TextureType::TEXTURE_2D)
                .def_readwrite("spec", &atcg::FramebufferTextureSpecification::spec)
                .def_readwrite("type", &atcg::FramebufferTextureSpecification::type);

            m_framebuffer_specification.def(py::init<>())
                .def(py::init<>(
                         [](uint32_t width,
                            uint32_t height,
                            uint32_t num_samples,
                            const std::vector<atcg::FramebufferTextureSpecification>& attachements)
                         {
                             atcg::FramebufferSpecification spec;
                             spec.width        = width;
                             spec.height       = height;
                             spec.num_samples  = num_samples;
                             spec.attachements = attachements;
                             return spec;
                         }),
                     "width"_a,
                     "height"_a,
                     "num_samples"_a,
                     "attachements"_a)
                .def(py::init<>(
                         [](uint32_t width,
                            uint32_t height,
                            uint32_t depth,
                            uint32_t num_samples,
                            const std::vector<atcg::FramebufferTextureSpecification>& attachements)
                         {
                             atcg::FramebufferSpecification spec;
                             spec.width        = width;
                             spec.height       = height;
                             spec.depth        = depth;
                             spec.num_samples  = num_samples;
                             spec.attachements = attachements;
                             return spec;
                         }),
                     "width"_a,
                     "height"_a,
                     "depth"_a,
                     "num_samples"_a,
                     "attachements"_a)
                .def_readwrite("width", &atcg::FramebufferSpecification::width)
                .def_readwrite("height", &atcg::FramebufferSpecification::height)
                .def_readwrite("depth", &atcg::FramebufferSpecification::depth)
                .def_readwrite("num_samples", &atcg::FramebufferSpecification::num_samples)
                .def_readwrite("attachements",
                               &atcg::FramebufferSpecification::attachements,
                               "The attachement specifications. Returns a copy; assign a new list instead of "
                               "appending to it.");

            m_framebuffer.def(py::init<>())
                .def(py::init<uint32_t, uint32_t>(), "width"_a, "height"_a)
                .def_static("create", &atcg::Framebuffer::create, "specification"_a)
                .def("attachColor", &atcg::Framebuffer::attachColor)
                .def("attachColorMultiSample", &atcg::Framebuffer::attachColorMultiSample, "num_samples"_a)
                .def("attachTexture", &atcg::Framebuffer::attachTexture, "texture"_a)
                .def("attachCubeFace",
                     &atcg::Framebuffer::attachCubeFace,
                     "cube_map"_a,
                     "face_index"_a,
                     "mip_level"_a = 0)
                .def("attachDepth", py::overload_cast<>(&atcg::Framebuffer::attachDepth))
                .def("attachDepth",
                     py::overload_cast<const atcg::ref_ptr<atcg::Texture>&>(&atcg::Framebuffer::attachDepth),
                     "depth_map"_a)
                .def("attachDepthMultiSample", &atcg::Framebuffer::attachDepthMultiSample, "num_samples"_a)
                .def("detachColor", &atcg::Framebuffer::detachColor)
                .def("detachDepth", &atcg::Framebuffer::detachDepth)
                .def("detachAllAttachements", &atcg::Framebuffer::detachAllAttachements)
                .def("blit", &atcg::Framebuffer::blit, "source"_a, "color"_a = true, "depth"_a = true)
                .def(
                    "getColorAttachement",
                    [](const atcg::ref_ptr<atcg::Framebuffer>& self,
                       const uint32_t slot) -> atcg::ref_ptr<atcg::Texture>
                    {
                        if(slot >= self->numColorAttachements())
                        {
                            ATCG_ERROR("Framebuffer: color attachement slot {0} is out of range ({1} attachements)",
                                       slot,
                                       self->numColorAttachements());
                            return nullptr;
                        }
                        return self->getColorAttachement(slot);
                    },
                    "slot"_a = 0)
                .def("numColorAttachements", &atcg::Framebuffer::numColorAttachements)
                .def("getNumberAttachements", &atcg::Framebuffer::getNumberAttachements)
                .def("getDepthAttachement", &atcg::Framebuffer::getDepthAttachement)
                .def("getID", &atcg::Framebuffer::getID)
                .def("width", &atcg::Framebuffer::width)
                .def("height", &atcg::Framebuffer::height)
                .def_static("currentFramebuffer", &atcg::Framebuffer::currentFramebuffer)
                .def("complete", &atcg::Framebuffer::complete);

            m_skybox.def(py::init<>())
                .def(py::init<const atcg::ref_ptr<atcg::Texture2D>&>(), "skybox_texture"_a)
                .def("setSkyboxTexture", &atcg::Skybox::setSkyboxTexture, "skybox_texture"_a)
                .def("getSkyboxTexture", &atcg::Skybox::getSkyboxTexture)
                .def("getSkyboxCubeMap", &atcg::Skybox::getSkyboxCubeMap)
                .def("getIrradianceMap", &atcg::Skybox::getIrradianceMap)
                .def("getPrefilteredMap", &atcg::Skybox::getPrefilteredMap);

            m_vertex_buffer.def(py::init<>())
                .def(py::init<size_t>(), "size"_a)
                .def(py::init(
                         [](py::buffer buffer)
                         {
                             py::buffer_info info = buffer.request();
                             return atcg::make_ref<atcg::VertexBuffer>(info.ptr, info.size * info.itemsize);
                         }),

                     "data"_a)
                .def(
                    "setData",
                    [](const atcg::ref_ptr<atcg::VertexBuffer>& vbo, py::buffer buffer)
                    {
                        py::buffer_info info = buffer.request();
                        vbo->setData(info.ptr, info.size * info.itemsize);
                    },
                    "data"_a)
                .def("resize", &atcg::VertexBuffer::resize, "size"_a)
                .def("clone", &atcg::VertexBuffer::clone)
                .def("getDevicePointer",
                     [](const atcg::ref_ptr<atcg::VertexBuffer>& self)
                     { return reinterpret_cast<uintptr_t>(self->getDevicePointer()); })
                .def("getHostPointer",
                     [](const atcg::ref_ptr<atcg::VertexBuffer>& self)
                     { return reinterpret_cast<uintptr_t>(self->getHostPointer()); })
                .def("unmapHostPointers", &atcg::VertexBuffer::unmapHostPointers)
                .def("unmapDevicePointers", &atcg::VertexBuffer::unmapDevicePointers)
                .def("unmapPointers", &atcg::VertexBuffer::unmapPointers)
                .def("isDeviceMapped", &atcg::VertexBuffer::isDeviceMapped)
                .def("isHostMapped", &atcg::VertexBuffer::isHostMapped)
                .def("getLayout", &atcg::VertexBuffer::getLayout)
                .def("setLayout", &atcg::VertexBuffer::setLayout, "layout"_a)
                .def("size", &atcg::VertexBuffer::size)
                .def("capacity", &atcg::VertexBuffer::capacity)
                .def("ID", &atcg::VertexBuffer::ID);

            m_index_buffer.def(py::init<>())
                .def(py::init<size_t>(), "count"_a)
                .def(py::init(
                         [](py::array_t<uint32_t, py::array::c_style | py::array::forcecast> indices) {
                             return atcg::make_ref<atcg::IndexBuffer>(indices.data(),
                                                                      static_cast<size_t>(indices.size()));
                         }),
                     "indices"_a)
                .def(
                    "setData",
                    [](const atcg::ref_ptr<atcg::IndexBuffer>& self,
                       py::array_t<uint32_t, py::array::c_style | py::array::forcecast> indices)
                    { self->setData(indices.data(), static_cast<size_t>(indices.size())); },
                    "indices"_a)
                .def("getCount", &atcg::IndexBuffer::getCount);

            m_vertex_array.def(py::init<>())
                .def("pushVertexBuffer", &atcg::VertexArray::pushVertexBuffer, "vbo"_a)
                .def("popVertexBuffer", &atcg::VertexArray::popVertexBuffer)
                .def("peekVertexBuffer", &atcg::VertexArray::peekVertexBuffer)
                .def("setIndexBuffer", &atcg::VertexArray::setIndexBuffer, "ibo"_a)
                .def("pushInstanceBuffer", &atcg::VertexArray::pushInstanceBuffer, "vbo"_a)
                .def("getIndexBuffer", &atcg::VertexArray::getIndexBuffer);

            m_buffer_layout.def(py::init<>())
                .def(py::init<const std::vector<atcg::BufferElement>&>(), "elements"_a)
                .def("getStride", &atcg::BufferLayout::getStride)
                .def("getElements", &atcg::BufferLayout::getElements)
                .def(
                    "__iter__",
                    [](const atcg::BufferLayout& self) { return py::make_iterator(self.begin(), self.end()); },
                    py::keep_alive<0, 1>())
                .def("__len__", [](const atcg::BufferLayout& self) { return self.getElements().size(); });

            m_buffer_element.def(py::init<>())
                .def(py::init<atcg::ShaderDataType, const std::string&, bool>(),
                     "type"_a,
                     "name"_a,
                     "normalized"_a = false)
                .def_readwrite("name", &atcg::BufferElement::name)
                .def_readwrite("type", &atcg::BufferElement::type)
                .def_readwrite("size", &atcg::BufferElement::size)
                .def_readwrite("offset", &atcg::BufferElement::offset)
                .def_readwrite("normalized", &atcg::BufferElement::normalized)
                .def("getComponentCount", &atcg::BufferElement::getComponentCount);

            m_shader_data_type.value("None", atcg::ShaderDataType::None)
                .value("Float", atcg::ShaderDataType::Float)
                .value("Float2", atcg::ShaderDataType::Float2)
                .value("Float3", atcg::ShaderDataType::Float3)
                .value("Float4", atcg::ShaderDataType::Float4)
                .value("Mat3", atcg::ShaderDataType::Mat3)
                .value("Mat4", atcg::ShaderDataType::Mat4)
                .value("Int", atcg::ShaderDataType::Int)
                .value("Int2", atcg::ShaderDataType::Int2)
                .value("Int3", atcg::ShaderDataType::Int3)
                .value("Int4", atcg::ShaderDataType::Int4)
                .value("Bool", atcg::ShaderDataType::Bool)
                .export_values();

            m_compile_data
                .def(py::init<>(
                         [](uint32_t num_samples)
                         {
                             atcg::CompileData data;
                             data.num_samples = num_samples;
                             return data;
                         }),
                     "num_samples"_a = 1)
                .def_readwrite("num_samples", &atcg::CompileData::num_samples);

            m_render_pass.def("name", &atcg::RenderPass::name);

            m_render_graph.def(py::init<>())
                .def("addRenderPass", &atcg::RenderGraph::addRenderPass, "pass"_a)
                .def("addDependency",
                     &atcg::RenderGraph::addDependency,
                     "source"_a,
                     "source_name"_a,
                     "target"_a,
                     "target_name"_a)
                .def("compile", &atcg::RenderGraph::compile, "compile_data"_a)
                .def("ensureCompiled", &atcg::RenderGraph::ensureCompiled, "compile_data"_a)
                .def("isCompiled", &atcg::RenderGraph::isCompiled)
                .def("exportToDOT", &atcg::RenderGraph::exportToDOT, "path"_a)
                .def("outputPass",
                     [](const atcg::ref_ptr<atcg::RenderGraph>& self) -> atcg::ref_ptr<atcg::RenderPass>
                     { return self->outputPass(); })
                .def("outputPassHandle", &atcg::RenderGraph::outputPassHandle)
                .def("setOutputFramebuffer", &atcg::RenderGraph::setOutputFramebuffer, "fbo"_a)
                .def("setMaxCachedResolutions",
                     &atcg::RenderGraph::setMaxCachedResolutions,
                     "max_cached_resolutions"_a);

            m_render_pass_registry
                .def(
                    "createRenderPass",
                    [](const std::string& type, atcg::Dictionary& properties) -> atcg::ref_ptr<atcg::RenderPass>
                    {
                        auto registry =
                            atcg::SystemRegistry::instance()->getSystem<atcg::RenderPassRegistry::Registry>();
                        if(!registry)
                        {
                            ATCG_ERROR("RenderPass registry not found");
                            return nullptr;
                        }
                        return atcg::RenderPassRegistry::createRenderPass(registry, type, properties);
                    },
                    "type"_a,
                    "properties"_a)
                .def(
                    "createRenderPass",
                    [](const std::string& type) -> atcg::ref_ptr<atcg::RenderPass>
                    {
                        auto registry =
                            atcg::SystemRegistry::instance()->getSystem<atcg::RenderPassRegistry::Registry>();
                        if(!registry)
                        {
                            ATCG_ERROR("RenderPass registry not found");
                            return nullptr;
                        }
                        atcg::Dictionary properties;
                        return atcg::RenderPassRegistry::createRenderPass(registry, type, properties);
                    },
                    "type"_a)
                .def("getRegisteredTypes",
                     []() -> std::vector<std::string>
                     {
                         auto registry =
                             atcg::SystemRegistry::instance()->getSystem<atcg::RenderPassRegistry::Registry>();
                         if(!registry)
                         {
                             ATCG_ERROR("RenderPass registry not found");
                             return {};
                         }
                         return registry->getRegisteredTypes();
                     });
        });
}
}    // namespace python
}    // namespace atcg
