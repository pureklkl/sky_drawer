
// TODO: Add OpenCL kernel code here.
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void skyFilter(__read_only image2d_t supersample, 					 
					 __constant size_t *sampleWH, 
					 __constant float *sampleData,
					 __write_only image2d_t output){ 
	const int2 pos = {get_global_id(0), get_global_id(1)};
	float4 pixel = {0.f, 0.f,0.f,0.f};
	for(size_t i = 0; i < sampleWH[0]; i++){ 
		for(size_t j = 0; j < sampleWH[1]; j++){ 
			pixel = pixel + read_imagef(supersample, sampler, (int2)(pos.x*sampleWH[0]+i, pos.y*sampleWH[1]+j));
		}
	}
#if 1
	// Apply tone mapping function
	pixel = select((float4)(1.0f, 1.0f, 1.0f, 0.f) - exp(-pixel) , pow(pixel*0.38317f, 1.0f/2.2f), pixel<1.413f);
#endif
	pixel = min(pixel, (float4)(1.0f, 1.0f, 1.0f, 0.f))*255.f;
	write_imageui( output, pos,  convert_uint4(pixel));
}