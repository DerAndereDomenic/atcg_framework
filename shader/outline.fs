#version 400 core

layout(location = 0) out vec4 FragColor;

in vec2 frag_uv;

uniform int selected_entity_id;
uniform isampler2D selection_mask;
uniform sampler2D input_color_buffer;

const int OUTLINE_THICKNESS = 3;
const vec3 outline_color = vec3(1.0, 0.69, 0.0);

void main()
{
    FragColor = texture(input_color_buffer, frag_uv);

    if(selected_entity_id == -1)
    {
        return;
    }

    // The mask and the output have the same resolution (both FULL_FRAMEBUFFER)
    ivec2 size  = textureSize(selection_mask, 0);
    ivec2 pixel = ivec2(gl_FragCoord.xy);

    // Pixels covered by the selected entity keep their color; the outline is drawn outside the silhouette
    if(texelFetch(selection_mask, pixel, 0).r == selected_entity_id)
    {
        return;
    }

    for(int x = -OUTLINE_THICKNESS; x <= OUTLINE_THICKNESS; x++)
    {
        for(int y = -OUTLINE_THICKNESS; y <= OUTLINE_THICKNESS; y++)
        {
            ivec2 sample_pixel = clamp(pixel + ivec2(x, y), ivec2(0), size - 1);
            if(texelFetch(selection_mask, sample_pixel, 0).r == selected_entity_id)
            {
                FragColor = vec4(outline_color, 1);
                return;
            }
        }
    }
}
