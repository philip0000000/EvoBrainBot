struct FragmentInput
{
    float4 Position : SV_Position;
    float2 Local : TEXCOORD0;
    float4 Color : TEXCOORD1;
    nointerpolation float Shape : TEXCOORD2;
};

// Shades rectangles directly and antialiases circle edges in screen space.
float4 main(FragmentInput input) : SV_Target0
{
    float coverage = 1.0f;
    if (input.Shape > 0.5f) {
        const float signed_distance = length(input.Local) - 1.0f;
        const float edge_width = max(fwidth(signed_distance), 0.0001f);
        coverage = 1.0f - smoothstep(-edge_width, edge_width, signed_distance);
    }
    return float4(input.Color.rgb, input.Color.a * coverage);
}
