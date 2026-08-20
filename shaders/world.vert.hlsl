struct VertexInput
{
    float2 Center : TEXCOORD0;
    float2 HalfSize : TEXCOORD1;
    float2 Rotation : TEXCOORD2;
    float4 Color : TEXCOORD3;
    float Shape : TEXCOORD4;
};

struct VertexOutput
{
    float4 Position : SV_Position;
    float2 Local : TEXCOORD0;
    float4 Color : TEXCOORD1;
    nointerpolation float Shape : TEXCOORD2;
};

cbuffer ViewportUniforms : register(b0, space1)
{
    float2 TargetSize;
    float2 Padding;
};

static const uint QuadIndices[6] = {0, 1, 2, 0, 2, 3};
static const float2 QuadCorners[4] = {
    {-1.0f, -1.0f},
    { 1.0f, -1.0f},
    { 1.0f,  1.0f},
    {-1.0f,  1.0f}
};

// Expands one compact shape instance into a rotated screen-space quad.
VertexOutput main(VertexInput input, uint vertex_id : SV_VertexID)
{
    const float2 local = QuadCorners[QuadIndices[vertex_id]];
    const float2 unrotated = local * input.HalfSize;
    const float2 rotated = float2(
        unrotated.x * input.Rotation.x - unrotated.y * input.Rotation.y,
        unrotated.x * input.Rotation.y + unrotated.y * input.Rotation.x);
    const float2 pixel = input.Center + rotated;

    VertexOutput output;
    output.Position = float4(
        pixel.x / TargetSize.x * 2.0f - 1.0f,
        1.0f - pixel.y / TargetSize.y * 2.0f,
        0.0f,
        1.0f);
    output.Local = local;
    output.Color = input.Color;
    output.Shape = input.Shape;
    return output;
}
