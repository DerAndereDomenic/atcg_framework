#version 400 core

layout(location = 0) out vec4 FragColor;

in vec2 frag_uv;

uniform int active_entity_id;
uniform isampler2D selection_mask;
uniform sampler2D input_color_buffer;

const int OUTLINE_THICKNESS = 3;
const vec3 active_outline_color = vec3(1.0, 0.69, 0.0);
const vec3 selected_outline_color = vec3(0.8, 0.35, 0.0);

void main()
{
    FragColor = texture(input_color_buffer, frag_uv);

    // Nothing is selected
    if(active_entity_id == -1)
    {
        return;
    }

    // The mask and the output have the same resolution (both FULL_FRAMEBUFFER)
    ivec2 size  = textureSize(selection_mask, 0);
    ivec2 pixel = ivec2(gl_FragCoord.xy);

    // Pixels covered by any selected entity keep their color; the outline is drawn outside the silhouettes
    if(texelFetch(selection_mask, pixel, 0).r != -1)
    {
        return;
    }

    bool found_selected = false;
    for(int x = -OUTLINE_THICKNESS; x <= OUTLINE_THICKNESS; x++)
    {
        for(int y = -OUTLINE_THICKNESS; y <= OUTLINE_THICKNESS; y++)
        {
            ivec2 sample_pixel = clamp(pixel + ivec2(x, y), ivec2(0), size - 1);
            int id             = texelFetch(selection_mask, sample_pixel, 0).r;
            if(id == active_entity_id)
            {
                FragColor = vec4(active_outline_color, 1);
                return;
            }
            found_selected = found_selected || id != -1;
        }
    }

    if(found_selected)
    {
        FragColor = vec4(selected_outline_color, 1);
    }
}
