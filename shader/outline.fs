#version 400 core

layout(location = 0) out vec4 FragColor;

in vec2 frag_uv;

uniform int selected_entity_id;
uniform isampler2D entity_ids;
uniform sampler2D input_color_buffer;

// vec3 outline_color = vec3(1, 0, 0);

const int OUTLINE_THICKNESS = 2;
const vec3 outline_color = vec3(1.0, 0.69, 0.0);

void main()
{
    FragColor = texture(input_color_buffer, frag_uv);

    if(selected_entity_id == -1)
    {
        return;
    }

    ivec2 size = textureSize(entity_ids, 0);
    int center_id = texture(entity_ids, frag_uv).r;
    if(center_id != selected_entity_id)
    {
        return;
    }

    float dx = 1.0 / float(size.x);
    float dy = 1.0 / float(size.y);

    for(int x = -OUTLINE_THICKNESS; x <= OUTLINE_THICKNESS; x++)
    {
        for(int y = -OUTLINE_THICKNESS; y <= OUTLINE_THICKNESS; y++)
        {
            int sample_id = texture(entity_ids, frag_uv + vec2(x * dx, y * dy)).r;
            if(sample_id != selected_entity_id)
            {
                FragColor = vec4(outline_color, 1);
                return;
            }
        }
    }
}