
// TODO: Add OpenCL kernel code here.

__kernel void testCL(__write_only image2d_t supersample){ 
	const int2 pos = {get_global_id(0), get_global_id(1)};
	uint4 pixel = { pos.x%256,   pos.y%256, 0, 0 };
	write_imageui( supersample, pos, pixel);
}