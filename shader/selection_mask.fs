#version 330 core

layout(location = 0) out int outEntityID;

uniform int entityID;

void main()
{
    outEntityID = entityID;
}
