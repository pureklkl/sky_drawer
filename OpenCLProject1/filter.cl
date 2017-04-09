
// TODO: Add OpenCL kernel code here.
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void filter(__read_only image2d_t image, , __write_only image2d_t output){ 
	const int2 pos = {get_global_id(0), get_global_id(1)};
	float4 pixel = {0.f, 0.f,0.f,0.f};
	for(int i=0;i<4;i++){ 
		for(int j=0;j<4;j++){ 
			pixel = pixel + convert_float4(read_imageui(supersample, sampler, (int2)(pos.x*4+i, pos.y*4+j)));
		}
	}
	pixel = pixel/16.f;
	write_imageui( output, pos,  convert_uint4(pixel));
}