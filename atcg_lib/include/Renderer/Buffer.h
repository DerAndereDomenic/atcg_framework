#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace atcg
{
    /**
     * @brief This enum holds the possible vertex attributes that can be passed to a shader via a vertex buffer
     * 
     */
    enum class ShaderDataType
    {
        None = 0,
        Float,
        Float2,
        Float3,
        Float4,
        Mat3,
        Mat4,
        Int,
        Int2,
        Int3,
        Int4,
        Bool
    };

    /**
     * @brief Get the size of an element in byte
     * 
     * @param type The Data type
     * @return uint32_t The size in bytes
     */
    static uint32_t ShaderDataTypeSize(ShaderDataType type)
    {
        switch(type)
        {
            case ShaderDataType::Float:    return 4;
			case ShaderDataType::Float2:   return 4 * 2;
			case ShaderDataType::Float3:   return 4 * 3;
			case ShaderDataType::Float4:   return 4 * 4;
			case ShaderDataType::Mat3:     return 4 * 3 * 3;
			case ShaderDataType::Mat4:     return 4 * 4 * 4;
			case ShaderDataType::Int:      return 4;
			case ShaderDataType::Int2:     return 4 * 2;
			case ShaderDataType::Int3:     return 4 * 3;
			case ShaderDataType::Int4:     return 4 * 4;
			case ShaderDataType::Bool:     return 1;
        }
    };

    /**
     * @brief A class to model an element inside the buffer
     * 
     */
    struct BufferElement
    {
        std::string name;
        ShaderDataType type;
        uint32_t size;
        size_t offset;
        bool normalized;

        BufferElement() = default;

        BufferElement(ShaderDataType type, const std::string& name, bool normalized = false)
            :name(name), type(type), size(ShaderDataTypeSize(type)), offset(0), normalized(normalized)
        {}

        uint32_t getComponentCount() const
        {
            switch(type)
            {
                case ShaderDataType::Float:   return 1;
				case ShaderDataType::Float2:  return 2;
				case ShaderDataType::Float3:  return 3;
				case ShaderDataType::Float4:  return 4;
				case ShaderDataType::Mat3:    return 3; // 3* float3
				case ShaderDataType::Mat4:    return 4; // 4* float4
				case ShaderDataType::Int:     return 1;
				case ShaderDataType::Int2:    return 2;
				case ShaderDataType::Int3:    return 3;
				case ShaderDataType::Int4:    return 4;
				case ShaderDataType::Bool:    return 1;
            }
        }
    };

    /**
     * @brief The layout of the buffer.
     * Basically a collection of BufferElements
     */
    class BufferLayout
    {
    public:
        BufferLayout() {}

        BufferLayout(std::initializer_list<BufferElement> elements)
            :_elements(elements)
        {
            calculateOffsetAndStride();
        }

        uint32_t getStride() const {return _stride;}

        const std::vector<BufferElement>& getElements() const {return _elements;}

        std::vector<BufferElement>::iterator begin() { return _elements.begin(); }
		std::vector<BufferElement>::iterator end() { return _elements.end(); }
		std::vector<BufferElement>::const_iterator begin() const { return _elements.begin(); }
		std::vector<BufferElement>::const_iterator end() const { return _elements.end(); }

    private:
        void calculateOffsetAndStride()
        {
            size_t offset = 0;
            _stride = 0;
            for(auto& element : _elements)
            {
                element.offset = offset;
                offset += element.size;
                _stride += element.size;
            }
        }
    private:
        std::vector<BufferElement> _elements;
        uint32_t _stride = 0;
    };

    /**
     * @brief A class to model a vertex buffer
     */
    class VertexBuffer
    {
    public:
        /**
         * @brief Construct a new Vertex Buffer object
         */
        VertexBuffer() = default;

        /**
         * @brief Construct a new empty Vertex Buffer object
         * 
         * @param size The size of be allocated on the GPU
         */
        VertexBuffer(uint32_t size);

        /**
         * @brief Construct a new Vertex Buffer object
         * 
         * @param data The data to fill the buffer with
         * @param size The size
         */
        VertexBuffer(const void* data, uint32_t size);

        /**
         * @brief Destroy the Vertex Buffer object
         */
        ~VertexBuffer();

        /**
         * @brief Use this vbo
         */
        void use() const;

        /**
         * @brief Set the Data of the buffer
         * 
         * @param data The data
         * @param size The size
         */
        void setData(const void* data, uint32_t size);

        /**
         * @brief Get the Layout
         * 
         * @return const BufferLayout& The layout of the buffer
         */
        inline const BufferLayout& getLayout() const {return _layout;}

        /**
         * @brief Set the Layout
         * 
         * @param layout The layout
         */
        inline void setLayout(const BufferLayout& layout) {_layout = layout;}
    private:
        uint32_t _ID;
        BufferLayout _layout;
    };

    /**
     * @brief A class to model an IndexBuffer
     */
    class IndexBuffer
    {
    public:
        /**
         * @brief Construct a new Index Buffer object
         */
        IndexBuffer() = default;

        /**
         * @brief Construct a new Index Buffer object
         * 
         * @param data The data
         * @param count The size
         */
        IndexBuffer(const uint32_t* indices, uint32_t count);

        /**
         * @brief Destroy the Index Buffer object
         */
        ~IndexBuffer();

        /**
         * @brief Use this ibo
         */
        void use() const;

        /**
         * @brief Get the Count of objects
         * 
         * @return uint32_t The count
         */
        inline uint32_t getCount() const {return _count;}
    private:
        uint32_t _ID;
        uint32_t _count;
    };
}