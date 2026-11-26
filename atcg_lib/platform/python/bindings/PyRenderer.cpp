#include "Bindings.h"

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
// Pipeline states, graphics commands, renderer, shaders, textures and buffers
void bindRenderer(BindingContext& ctx)
{
    py::module_ m = ctx.module();

    auto m_renderer = m.def_submodule("Renderer");

    auto m_graphics_api    = m.def_submodule("GraphicsCommand");
    auto m_renderer_system = py::class_<atcg::RendererSystem, atcg::ref_ptr<atcg::RendererSystem>>(m, "RendererSystem");
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
    auto m_texture2d    = py::class_<atcg::Texture2D, atcg::Asset, atcg::ref_ptr<atcg::Texture2D>>(m, "Texture2D");
    auto m_texture_cube = py::class_<atcg::TextureCube, atcg::ref_ptr<atcg::TextureCube>>(m, "TextureCube");
    auto m_framebuffer  = py::class_<atcg::Framebuffer, atcg::ref_ptr<atcg::Framebuffer>>(m, "Framebuffer");

    auto m_vertex_buffer    = py::class_<atcg::VertexBuffer, atcg::ref_ptr<atcg::VertexBuffer>>(m, "VertexBuffer");
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
                .def("enableDepthTesting", &atcg::DepthState::enableDepthTesting, "enable"_a)
                .def("enableDepthWrite", &atcg::DepthState::enableDepthWrite, "enable"_a);

            m_blend_state.def(py::init<>())
                .def_readwrite("blend_enabled", &atcg::BlendState::blend_enabled)
                .def("enableBlending", &atcg::BlendState::enableBlending, "enable"_a);

            m_rasterizer_state.def(py::init<>())
                .def_readwrite("cull_mode", &atcg::RasterizerState::cull_mode)
                .def_readwrite("culling_enabled", &atcg::RasterizerState::culling_enabled)
                .def_readwrite("depth_state", &atcg::RasterizerState::depth_state)
                .def_readwrite("blend_state", &atcg::RasterizerState::blend_state)
                .def_readwrite("point_size", &atcg::RasterizerState::point_size)
                .def_readwrite("line_size", &atcg::RasterizerState::line_size)
                .def("setCullMode", &atcg::RasterizerState::setCullMode, "mode"_a)
                .def("enableCulling", &atcg::RasterizerState::enableCulling, "enable"_a)
                .def("setPointSize", &atcg::RasterizerState::setPointSize, "size"_a)
                .def("setLineSize", &atcg::RasterizerState::setLineSize, "size"_a)
                .def("setDepthState", &atcg::RasterizerState::setDepthState, "depth_state"_a)
                .def("setBlendState", &atcg::RasterizerState::setBlendState, "blend_state"_a);

            m_graphics_pipeline.def(py::init<>())
                .def_readwrite("shader", &atcg::GraphicsPipeline::shader)
                .def_readwrite("rasterizer_state", &atcg::GraphicsPipeline::rasterizer_state)
                .def_readwrite("primitive_topology", &atcg::GraphicsPipeline::primitive_topology)
                .def("setShader", &atcg::GraphicsPipeline::setShader, "shader"_a)
                .def("setRasterizerState", &atcg::GraphicsPipeline::setRasterizerState, "rasterizer_state"_a)
                .def("setPrimitiveTopology", &atcg::GraphicsPipeline::setPrimitiveTopology, "topology"_a);

            m_graphics_api.def("beginRenderPass", &atcg::GraphicsCommand::beginRenderPass, "target_framebuffer"_a)
                .def("endRenderPass", &atcg::GraphicsCommand::endRenderPass)
                .def("clear", &atcg::GraphicsCommand::clear)
                .def("bindTexture", &atcg::GraphicsCommand::bindTexture, "slot"_a, "texture"_a)
                .def("bindStorageBuffer", &atcg::GraphicsCommand::bindStorageBuffer, "slot"_a, "buffer"_a)
                .def("setClearColor", &atcg::GraphicsCommand::setClearColor, "color"_a);

            m_renderer.def("init", &atcg::Renderer::init)
                .def("finishFrame", &atcg::Renderer::finishFrame)
                .def(
                    "drawCADGrid",
                    [](const atcg::ref_ptr<atcg::PerspectiveCamera>& camera) { atcg::Renderer::drawCADGrid(camera); },
                    "camera"_a)
                .def("drawCircle",
                     &atcg::Renderer::drawCircle,
                     "position"_a,
                     "radius"_a,
                     "thickness"_a,
                     "color"_a,
                     "camera"_a,
                     "entity_id"_a)
                .def(
                    "drawImage",
                    [](const atcg::ref_ptr<atcg::Framebuffer>& img) { atcg::Renderer::drawImage(img); },
                    "img"_a)
                .def(
                    "drawImage",
                    [](const atcg::ref_ptr<atcg::Texture2D>& img) { atcg::Renderer::drawImage(img); },
                    "img"_a)
                .def("getFramebuffer", &atcg::Renderer::getFramebuffer)
                .def("resize", &atcg::Renderer::resize)
                .def("getFrameCounter", &atcg::Renderer::getFrameCounter)
                .def("popTextureID", &atcg::Renderer::popTextureID)
                .def("pushTextureID", &atcg::Renderer::pushTextureID, "id"_a);

            m_renderer_system.def(py::init<>())
                .def("init", &atcg::RendererSystem::init)
                .def("finishFrame", &atcg::RendererSystem::finishFrame)
                .def(
                    "drawCADGrid",
                    [](const atcg::ref_ptr<atcg::RendererSystem>& self,
                       const atcg::ref_ptr<atcg::PerspectiveCamera>& camera) { self->drawCADGrid(camera); },
                    "camera"_a)
                .def("drawCircle",
                     &atcg::RendererSystem::drawCircle,
                     "position"_a,
                     "radius"_a,
                     "thickness"_a,
                     "color"_a,
                     "camera"_a,
                     "entity_id"_a)
                .def(
                    "drawImage",
                    [](const atcg::ref_ptr<atcg::RendererSystem>& self, const atcg::ref_ptr<atcg::Framebuffer>& img)
                    { self->drawImage(img); },
                    "img"_a)
                .def(
                    "drawImage",
                    [](const atcg::ref_ptr<atcg::RendererSystem>& self, const atcg::ref_ptr<atcg::Texture2D>& img)
                    { self->drawImage(img); },
                    "img"_a)
                .def("getFramebuffer", &atcg::RendererSystem::getFramebuffer)
                .def("resize", &atcg::RendererSystem::resize)
                .def("getFrameCounter", &atcg::RendererSystem::getFrameCounter)
                .def("popTextureID", &atcg::RendererSystem::popTextureID)
                .def("pushTextureID", &atcg::RendererSystem::pushTextureID, "id"_a);

            m_shader.def(py::init<std::string>(), "compute_path"_a)
                .def(py::init<std::string, std::string>(), "vertex_path"_a, "fragment_path"_a)
                .def(py::init<std::string, std::string, std::string>(),
                     "vertex_path"_a,
                     "fragment_path"_a,
                     "geometry_path"_a)
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
                .def("setVec3", &atcg::Shader::setVec3, "uniform_name"_a, "value"_a)
                .def("setVec4", &atcg::Shader::setVec4, "uniform_name"_a, "value"_a)
                .def("setMat4", &atcg::Shader::setMat4, "uniform_name"_a, "value"_a)
                .def("setMVP", &atcg::Shader::setMVP, "model"_a, "view"_a, "projection"_a)
                .def("dispatch", &atcg::Shader::dispatch, "work_groups"_a)
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
                .def("onUpdate", &atcg::ShaderManager::onUpdate);

            m_shader_manager_system.def("getShader", &atcg::ShaderManagerSystem::getShader, "name"_a)
                .def("addShader", &atcg::ShaderManagerSystem::addShader, "name"_a, "shader"_a)
                .def("addShaderFromName", &atcg::ShaderManagerSystem::addShaderFromName, "name"_a)
                .def("addComputeShaderFromName", &atcg::ShaderManagerSystem::addComputeShaderFromName, "name"_a)
                .def("hasShader", &atcg::ShaderManagerSystem::hasShader, "name"_a)
                .def("onUpdate", &atcg::ShaderManagerSystem::onUpdate);

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
                .value("CLAMP_TO_EDGE", atcg::TextureWrapMode::CLAMP_TO_EDGE);

            m_texture_filter_mode.value("NEAREST", atcg::TextureFilterMode::NEAREST)
                .value("LINEAR", atcg::TextureFilterMode::LINEAR);

            m_texture_sampler.def(py::init<>())
                .def(py::init<>(
                         [](atcg::TextureFilterMode filter_mode, atcg::TextureWrapMode wrap_mode)
                         {
                             atcg::TextureSamplerSpecification sampler;
                             sampler.filter_mode = filter_mode;
                             sampler.wrap_mode   = wrap_mode;
                             return sampler;
                         }),
                     py::arg_v("filter_mode", atcg::TextureFilterMode::LINEAR, "linear"),
                     py::arg_v("wrap_mode", atcg::TextureWrapMode::REPEAT, "size"))
                .def_readwrite("wrap_mode", &atcg::TextureSamplerSpecification::wrap_mode)
                .def_readwrite("filter_mode", &atcg::TextureSamplerSpecification::filter_mode);

            m_texture_specification.def(py::init<>())
                .def(py::init<>(
                         [](uint32_t width,
                            uint32_t height,
                            uint32_t depth,
                            atcg::TextureSamplerSpecification sampler,
                            atcg::TextureFormat format)
                         {
                             atcg::TextureSpecification spec;
                             spec.width   = width;
                             spec.height  = height;
                             spec.depth   = depth;
                             spec.sampler = sampler;
                             spec.format  = format;
                             return spec;
                         }),
                     "width"_a,
                     "height"_a,
                     "depth"_a,
                     "sampler"_a,
                     "format"_a)
                .def_readwrite("width", &atcg::TextureSpecification::width)
                .def_readwrite("height", &atcg::TextureSpecification::height)
                .def_readwrite("depth", &atcg::TextureSpecification::depth)
                .def_readwrite("sampler", &atcg::TextureSpecification::sampler)
                .def_readwrite("format", &atcg::TextureSpecification::format);

            m_image.def(py::init<>())
                .def("load", &atcg::Image::load)
                .def("store", &atcg::Image::store)
                .def("applyGamma", &atcg::Image::applyGamma)
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

            m.def("imread", &atcg::IO::imread);
            m.def("imwrite", &atcg::IO::imwrite);

            m_texture2d
                .def_static(
                    "create",
                    [](atcg::TextureSpecification spec) { return atcg::Texture2D::create(spec); },
                    "specification"_a)
                .def_static(
                    "create",
                    [](const atcg::ref_ptr<atcg::Image>& img, atcg::TextureSpecification spec)
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
                .def("getID", &atcg::Texture2D::getID)
                .def(
                    "setData",
                    [](const atcg::ref_ptr<atcg::Texture2D>& texture, const torch::Tensor& data)
                    { texture->setData(data); },
                    "data"_a)
                .def("__setitem__",
                     [](const atcg::ref_ptr<atcg::Texture2D>& texture, py::slice idx, const torch::Tensor& t)
                     { texture->setData(t); })
                //.def("setData", [](const atcg::ref_ptr<atcg::Texture2D>& texture, const
                // atcg::ref_ptr<atcg::PixelUnpackBuffer>& data) {texture->setData(data);}, "data"_a)
                .def("getData", &atcg::Texture2D::getData);

            m_texture_cube
                .def_static(
                    "create",
                    [](atcg::TextureSpecification spec) { return atcg::TextureCube::create(spec); },
                    "specification"_a)
                .def_static(
                    "create",
                    [](const torch::Tensor& img) { return atcg::TextureCube::create(img); },
                    "img"_a)
                .def("getID", &atcg::TextureCube::getID)
                .def(
                    "setData",
                    [](const atcg::ref_ptr<atcg::TextureCube>& texture, const torch::Tensor& data)
                    { texture->setData(data); },
                    "data"_a)
                //.def("setData", [](const atcg::ref_ptr<atcg::TextureCube>& texture, const
                // atcg::ref_ptr<atcg::PixelUnpackBuffer>& data) {texture->setData(data);}, "data"_a)
                .def("getData", &atcg::TextureCube::getData);

            m_framebuffer.def(py::init<>())
                .def(py::init<uint32_t, uint32_t>())
                .def("attachColor", &atcg::Framebuffer::attachColor)
                .def("attachColorMultiSample", &atcg::Framebuffer::attachColorMultiSample)
                .def("attachTexture", &atcg::Framebuffer::attachTexture)
                .def("attachDepth", [](const atcg::ref_ptr<atcg::Framebuffer>& fbo) { fbo->attachDepth(); })
                .def("attachDepthMultiSample", &atcg::Framebuffer::attachDepthMultiSample)
                .def("blit", &atcg::Framebuffer::blit)
                .def("getColorAttachement", &atcg::Framebuffer::getColorAttachement)
                .def("getDepthAttachement", &atcg::Framebuffer::getDepthAttachement)
                .def("getID", &atcg::Framebuffer::getID)
                .def("width", &atcg::Framebuffer::width)
                .def("height", &atcg::Framebuffer::height)
                .def("currentFramebuffer", &atcg::Framebuffer::currentFramebuffer)
                .def("complete", &atcg::Framebuffer::complete);

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
                .def("getLayout", &atcg::VertexBuffer::getLayout)
                .def("setLayout", &atcg::VertexBuffer::setLayout, "setLayout"_a)
                .def("size", &atcg::VertexBuffer::size)
                .def("capacity", &atcg::VertexBuffer::capacity)
                .def("ID", &atcg::VertexBuffer::ID);

            m_buffer_layout.def(py::init<>())
                .def(py::init<const std::vector<atcg::BufferElement>&>(), "elements_a")
                .def("getStride", &atcg::BufferLayout::getStride)
                .def("getElements", &atcg::BufferLayout::getElements);

            m_buffer_element.def(py::init<>())
                .def(py::init<atcg::ShaderDataType, std::string>(), "type"_a, "name"_a)
                .def("getStgetComponentCountride", &atcg::BufferElement::getComponentCount);

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
        });
}
}    // namespace python
}    // namespace atcg
