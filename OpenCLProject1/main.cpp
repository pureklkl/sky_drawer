#include<iostream>
#include<string>
#include <chrono>
#include<sstream>
#include <algorithm>
#include <fstream>
#include <random>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "Utils.h"
#include "nishita_sky.h"

#ifndef M_PI
#define M_PI (3.14159265358979323846f)
#endif 

#define MAX_SOURCE_SIZE (0x100000)
#define MEM_SIZE (128)

#define OPENCL_INFO_SIZE 1024
#define DEFAULT_PLATFORM 0
#define DEFAULT_DEVICE 0
#define USE_DEVICE_NUM 1
#define SAMPLE_LENGTH 3

const float fov = 65.f;

class OpenclGPUPlatform {
public:
	static OpenclGPUPlatform& getOpenclPlatform() {
		static OpenclGPUPlatform instance;
		return instance;
	}
	void initialize(bool verbose) {
		cl_int ret;
		
		//platform 
		ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
		checkfatalError("Get platform num fail", ret);
		if (ret_num_platforms <= 0) { fatalError("Cannot find opencl platform"); }
		platform_id = new cl_platform_id[ret_num_platforms];
		ret = clGetPlatformIDs(ret_num_platforms, platform_id, &ret_num_platforms);
		checkfatalError("Get platform fail", ret);
		if (verbose) { showPlatform(); }

		//device
		ret = clGetDeviceIDs(platform_id[DEFAULT_PLATFORM], CL_DEVICE_TYPE_GPU, 0, NULL, &ret_num_devices);
		checkfatalError("Get device num fail", ret);
		if (ret_num_devices <= 0) { fatalError("Cannot find device"); }
		device_id = new cl_device_id[ret_num_devices];
		ret = clGetDeviceIDs(platform_id[DEFAULT_PLATFORM], CL_DEVICE_TYPE_GPU, ret_num_devices, device_id, &ret_num_devices);
		checkfatalError("Get device fail", ret);
		if (verbose) { showDevices(); }
		setDevice();

		//context
		context = clCreateContext(NULL, USE_DEVICE_NUM, &device_id[DEFAULT_DEVICE], NULL, NULL, &ret);
		checkfatalError("Create context fail", ret);
		if (verbose) { lineMsg("Create context sucess"); }

		//command queue
		command_queue = clCreateCommandQueue(context, device_id[DEFAULT_DEVICE], CL_QUEUE_PROFILING_ENABLE, &ret);
		checkfatalError("Create command queue fail", ret);
		if (verbose) { lineMsg("Create command queue sucess\nIntialize finished sucessful\n"); }
	}
	cl_context getContext() {
		return context;
	}
	cl_device_id getDeviceID() {
		return device_id[DEFAULT_DEVICE];
	}
	cl_command_queue getCommandQueue(){
		return command_queue;
	}
	cl_uint getGroupSize(){
		return maxGroupSize;
	}
private:
	OpenclGPUPlatform() :platform_id(NULL),device_id(NULL),context(NULL), command_queue(NULL) {};
	~OpenclGPUPlatform() {
		if (platform_id != NULL) { delete[] platform_id; }
		if (device_id != NULL) { delete[] device_id; }
		if (context != NULL) { clReleaseContext(context); }
		if (command_queue != NULL){ clReleaseCommandQueue(command_queue); }
	};
	void setDevice() {
		clGetDeviceInfo(device_id[DEFAULT_DEVICE], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnit), &maxComputeUnit, NULL);
		clGetDeviceInfo(device_id[DEFAULT_DEVICE], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxGroupSize), &maxGroupSize, NULL);
	}
	void showPlatform() {
		std::stringstream st;
		for (int i = 0; i < ret_num_platforms;i++) {
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_NAME, OPENCL_INFO_SIZE, showInfo, NULL);
			st << "Platform " << i << ": " << showInfo <<std::endl;
		}
		st << "Use platform : " << DEFAULT_PLATFORM;
		lineMsg(st.str().c_str());
	}
	void showDevices() {
		std::stringstream st;
		for (int i = 0; i < ret_num_devices; i++) {
			showDevice(i, device_id[i]);
		}
		st << "Use device: " << DEFAULT_DEVICE;
		lineMsg(st.str().c_str());
	}
	void showDevice(int i, cl_device_id device_id) {
		std::stringstream st;
		st << "Device " << i << ": ";
		clGetDeviceInfo(device_id, CL_DEVICE_NAME, OPENCL_INFO_SIZE, showInfo, NULL);
		st << showInfo << std::endl;
		clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnit), &maxComputeUnit, NULL);
		st << "Compute unit num: " << maxComputeUnit << std::endl;
		clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxGroupSize), &maxGroupSize, NULL);
		st << "Process unit per compute unit: " << maxGroupSize <<std::endl;
		cl_uint dim;
		clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(dim), &dim, NULL);
		size_t *localDimSize = new size_t[dim];
		clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, dim * sizeof(localDimSize[0]), localDimSize, NULL);
		st << "Local group dim size: ";
		for (int i = 0; i < dim; i++) {
			st << localDimSize[i] << " ";
		}
		delete[] localDimSize;
		st << std::endl;
		cl_ulong localBufferSize;
		clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localBufferSize), &localBufferSize, NULL);
		st << "local memory size " << byte2Kb(localBufferSize) << std::endl;
		cl_ulong constBufferSize;
		clGetDeviceInfo(device_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(constBufferSize), &constBufferSize, NULL);
		st << "constant memory size " << byte2Kb(constBufferSize) << std::endl;
		cl_ulong globalBufferSize;
		clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalBufferSize), &globalBufferSize, NULL);
		st << "global memory size " << byte2Mb(globalBufferSize) << std::endl;
		cl_bool support;
		clGetDeviceInfo(device_id, CL_DEVICE_IMAGE_SUPPORT, sizeof(support), &support, NULL);
		st << "support image " << support << std::endl;
		if (support) {
			size_t imagewidth;
			clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(imagewidth), &imagewidth, NULL);
			st << "support max image width " << imagewidth << std::endl;
			cl_bool imageHeight;
			clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(imageHeight), &imageHeight, NULL);
			st << "support max image height" << imageHeight << std::endl;
		}
		lineMsg(st.str().c_str());
	}

	cl_platform_id *platform_id;
	cl_uint ret_num_platforms;
	cl_device_id *device_id;
	cl_uint ret_num_devices;
	cl_uint maxComputeUnit;
	cl_uint maxGroupSize;
	cl_context context;
	cl_command_queue command_queue;
	char showInfo[OPENCL_INFO_SIZE];
};

class SimpleKernel {
public:
	SimpleKernel(std::string filename[], int filenum):kernel(NULL), program(NULL) {
		source = new char*[filenum];
		sourceLength = new size_t[filenum];
		sourcenum = filenum;
		for (int i = 0; i < filenum;i++) {
			std::fstream kernelFile(filename[i]);
			if (kernelFile.good()) {
				std::string content(
					(std::istreambuf_iterator<char>(kernelFile)),
					std::istreambuf_iterator<char>()
				);
				sourceLength[i] = content.size();
				source[i] = new char[sourceLength[i]];
				memcpy(source[i], content.c_str(), sourceLength[i]);
			}
			else {
				fatalError("Failed to load kernel.");
			}
			kernelFile.close();
		}
	}
	~SimpleKernel() {
		for (int i = 0; i < sourcenum; i++) {
			delete[] source[i];
		}
		delete[] source;
		if(kernel!=NULL){ clReleaseKernel(kernel); }
		if(program!=NULL){ clReleaseProgram(program); }
	}
	cl_kernel getKernel(std::string filename) {
		cl_int ret, retInfo;
		size_t sp = filename.find_first_of(".", 0);
		kernelName.append(filename.substr(0, sp));
		lineMsg("Create Kernel:");
		lineMsg(kernelName.c_str());
		program = clCreateProgramWithSource(OpenclGPUPlatform::getOpenclPlatform().getContext(), sourcenum, (const char**)source, sourceLength, &ret);
		checkfatalError("Create Program fail", ret);
		cl_device_id device_id = OpenclGPUPlatform::getOpenclPlatform().getDeviceID();
		ret = clBuildProgram(program, USE_DEVICE_NUM, &device_id, (compileOpt+filename).c_str(), NULL, NULL);
		if (ret != CL_SUCCESS) {
			char *programLog;
			size_t logSize;
			clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
			programLog = (char*)calloc(logSize + 1, sizeof(char));
			clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, logSize + 1, programLog, &logSize);
			lineMsg(programLog);
			checkfatalError("Build Program fail", ret);
			free(programLog);
		}
		kernel = clCreateKernel(program, kernelName.c_str(), &ret);

		checkfatalError("Create kernel fail", ret);
		std::stringstream st;
		st << "kernel group size: ";
		clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kernelGroupSize), &kernelGroupSize, NULL);
		st << kernelGroupSize<<std::endl<<"perfered work group multiple: ";
		clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(kernelGroupSizeMultiple), &kernelGroupSizeMultiple, NULL);
		st << kernelGroupSizeMultiple;
		lineMsg(st.str().c_str());
		return kernel;
	}
	cl_kernel getKernel() {
		return kernel;
	}
	size_t getKernelGroupSize() {
		return kernelGroupSize;
	}
	size_t getKernelGroupSizeMultiple() {
		return kernelGroupSizeMultiple;
	}
	std::string getkernelName() {
		return kernelName;
	}
private:
	cl_program program;
	cl_kernel kernel;
	std::string compileOpt = "-cl-no-signed-zeros -g -s ";
	size_t kernelGroupSize;
	size_t kernelGroupSizeMultiple;

	std::string kernelName;
	char** source;
	size_t sourcenum;
	size_t* sourceLength;
	
	char showInfo[OPENCL_INFO_SIZE];
};

class OpenCLGPUManager {
public:
	OpenCLGPUManager(SimpleKernel& kernel):kernel(kernel) {}
	void genParameter(size_t height, size_t width) {
		size_t totalWork = height * width;
		maxGroupSize = std::min(kernel.getKernelGroupSize(), OpenclGPUPlatform::getOpenclPlatform().getGroupSize());
		if (kernel.getKernelGroupSizeMultiple() < maxGroupSize) {
			optSize = maxGroupSize - maxGroupSize%kernel.getKernelGroupSizeMultiple();
		}
		else {
			optSize = maxGroupSize;
		}
		std::stringstream st;
		st << "kernel: " << kernel.getkernelName() << std::endl;
		st << " max group size " << maxGroupSize << std::endl;
		st << " prefer group size " << optSize << std::endl;
		lineMsg(st.str().c_str());
		bool maxret = calgroupsize2d(width, height, maxGroupSize, maxGroupSize2D), 
			 optret = calgroupsize2d(width, height, optSize, optSize2D);
		if (!maxret) {
			lineMsg("Warning: cannot find suitable max group size dimension");
		}
		else {
			st.str("");
			st << "max group dimension: " << maxGroupSize2D[0] << " " << maxGroupSize2D[1] << std::endl;
			lineMsg(st.str().c_str());
		}
		if (!optret) {
			lineMsg("Warning: cannot find suitable perfer multiple size dimension");
		}
		else {
			st.str("");
			st << "perfer dimension: " << optSize2D[0] << " " << optSize2D[1] << std::endl;
			lineMsg(st.str().c_str());
		}
		if (!maxret&& !optret) {
			fatalError("cannot find suitable group size dimension");
		}
	}
	void setParamters(void* parameters[], size_t size) {
		for (int i = 0; i < size;i++) {
			cl_int ret = clSetKernelArg(kernel.getKernel(), i, sizeof(cl_mem), parameters[i]);
			checkfatalError("set parameter fail", ret);
		}
	}
	template<class T>
	void setValueParamtersV(T v, size_t pos) {
		cl_int ret = clSetKernelArg(kernel.getKernel(), pos, sizeof(T), &v);
		checkfatalError("set parameter fail", ret);
	}
	size_t* getMaxGroupSize2D() {
		return maxGroupSize2D;
	}
	size_t* getOptSize2D() {
		return optSize2D;
	}
	size_t getMaxGroupSize() {
		return maxGroupSize;
	}
	size_t getOptSize() {
		return optSize;
	}
private:
	//Decompose size to two factors that width%factor1 == 0 and height%factor2==0;
	//it may be fail in case the factors cannot be found and the function will return false
	bool calgroupsize2d(size_t width, size_t height, size_t size, size_t result[]) {
		int factor1 = 1, factor2 = size;
		while (factor1 <= size) {
			if (size%factor1 == 0) {
				factor2 = size / factor1;
				if (width%factor1 == 0 && height%factor2 == 0) {
					result[0] = factor1;
					result[1] = factor2;
					return true;
				}
			}
			factor1++;
		}
		return false;
	}
	size_t maxGroupSize2D[2];
	size_t optSize2D[2];
	size_t maxGroupSize;
	size_t optSize;
	SimpleKernel& kernel;
};

class SkyDrawer {
public:
	//sample: dx, dy, weight
	SkyDrawer(std::string skysource, std::string samplesource, size_t width, size_t height, float *samples, size_t sampleH, size_t sampleW,bool verbose):
		width(width),
		height(height),
		samples(samples),
		sampleW(sampleW),
		sampleH(sampleH),
		tableformat{CL_INTENSITY, CL_FLOAT} 
	{
		OpenclGPUPlatform::getOpenclPlatform().initialize(verbose);
		
		sskyk = new SimpleKernel(&skysource, 1);
		skyk = sskyk->getKernel(skysource);
		skym = new OpenCLGPUManager(*sskyk);

		ssamplek = new SimpleKernel(&samplesource, 1);
		samplek = ssamplek->getKernel(samplesource);
		samplem = new OpenCLGPUManager(*ssamplek);

		superSampleW = width * sampleW;
		superSampleH = height * sampleH;
		samplenum = sampleH * sampleW;

		globalSample[0] = superSampleW;
		globalSample[1] = superSampleH;
		globalOutput[0] = width;
		globalOutput[1] = height;
		globalOutput[2] = 1;

		imgBuffer = new unsigned char[width*height * 4];

		angle = std::tan(fov * M_PI / 180.f * 0.5f);
	}

	//put buffers and parameters to gpu
	//buffers: 
	//	super sample - texture 
	//	output - texture
	//parameters:
	//	light intense table R/M - texture image 2D array - RG - intense is stored as RM, RM, RM...
	//	sample data dim - w,h - buffer
	//	sapmle data - x,y,weight - buffer
	//	super sample image -  sampled image
	//	output dim - final image dim
	//  fov
	//	sun direction - put during drawing
	//  output - final image
	//DrawSky parameters: light intense table, sample dim, sapmle data, super sample image,  output dim, fov, sun direction.
	//Filter parameters: super sample, sample data dim, sample data, output
	//NOTICE: OpenCl2.0 support RG/RGBA channel with float datatype but 1.2 not
	void init() {
		cl_int ret;

		set2Dformat(tableformat, CL_RG, CL_FLOAT);
		set2DDesc(tableDesc, CL_MEM_OBJECT_IMAGE2D, at.getCnum(), at.getRnum());//row pitch auto calculated
		lightTable = clCreateImage(
			OpenclGPUPlatform::getOpenclPlatform().getContext(), 
			CL_MEM_READ_ONLY |CL_MEM_COPY_HOST_PTR|CL_MEM_HOST_NO_ACCESS ,
			&tableformat,
			&tableDesc,
			at.getIntegratedTable(), 
			&ret);
		checkfatalError("Create light intense table buffer fail", ret);

		size_t sampleDataWHA[3] = {sampleW, sampleH, sampleW*sampleH};
		sampleDataWH = clCreateBuffer(
			OpenclGPUPlatform::getOpenclPlatform().getContext(),
			CL_MEM_READ_ONLY |CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS,
			3 * sizeof(sampleDataWHA[0]), sampleDataWHA, &ret);
		checkfatalError("Create sample dim fail", ret);

		sampleData = clCreateBuffer(
			OpenclGPUPlatform::getOpenclPlatform().getContext(),
			CL_MEM_READ_ONLY |CL_MEM_COPY_HOST_PTR|CL_MEM_HOST_NO_ACCESS ,
			samplenum * SAMPLE_LENGTH * sizeof(samples[0]), samples, &ret);
		checkfatalError("Create sample table buffer fail", ret);

		set2Dformat(superSampleformat, CL_RGBA, CL_FLOAT);
		set2DDesc(superSampleDesc, CL_MEM_OBJECT_IMAGE2D, superSampleW, superSampleH);
		superSample = clCreateImage(
			OpenclGPUPlatform::getOpenclPlatform().getContext(),
			CL_MEM_READ_WRITE| CL_MEM_HOST_NO_ACCESS,
			&superSampleformat,
			&superSampleDesc,
			NULL,
			&ret);
		checkfatalError("Create super sample image buffer fail", ret);

		set2Dformat(outputformat, CL_RGBA, CL_UNSIGNED_INT8);
		set2DDesc(outputDesc, CL_MEM_OBJECT_IMAGE2D, width, height);
		output = clCreateImage(
			OpenclGPUPlatform::getOpenclPlatform().getContext(),
			CL_MEM_WRITE_ONLY|CL_MEM_HOST_READ_ONLY,
			&outputformat,
			&outputDesc,
			NULL,
			&ret);
		checkfatalError("Create output image buffer fail", ret);

		void* params[8];
		params[0] = (void *)&lightTable;
		params[1] = (void *)&sampleDataWH;
		params[2] = (void *)&sampleData;
		params[3] = (void *)&superSample;
		skym->genParameter(superSampleH, superSampleW);
		skym->setParamters(params, 4); //params are copied to gpu buffer, thus can be reused.
		cl_float2 gloutput_cl = { globalOutput[0], globalOutput[1] };
		skym->setValueParamtersV<cl_float2>(gloutput_cl, 4);
		skym->setValueParamtersV<float>(angle, 5);

		params[0] = (void *)&superSample;
		params[3] = (void *)&output;
		samplem->genParameter(height, width);
		samplem->setParamters(params, 4);
	}

	void draw(Vec3f sunDir) {
		draw(sunDir, "testout.ppm");
	}

	void draw(Vec3f sunDir, char filename[]) {
		cl_int ret;
		auto t0 = std::chrono::high_resolution_clock::now();
		cl_float4 cl_sunDir = {sunDir.x, sunDir.y, sunDir.z};
		skym->setValueParamtersV<cl_float4>(cl_sunDir, 6);
		cl_event skydrawEvent;
		ret = clEnqueueNDRangeKernel(
			OpenclGPUPlatform::getOpenclPlatform().getCommandQueue(),
			sskyk->getKernel(),
			2,
			NULL,
			globalSample,
			skym->getOptSize2D(), 0, NULL, &skydrawEvent);
		checkfatalError("Create super sample image fail", ret);

		cl_event sampleEvent;
		ret = clEnqueueNDRangeKernel(
			OpenclGPUPlatform::getOpenclPlatform().getCommandQueue(),
			ssamplek->getKernel(),
			2,
			NULL,
			globalOutput,
			samplem->getOptSize2D(), 1, &skydrawEvent, &sampleEvent);
		checkfatalError("sample fail", ret);

		const static size_t origin[3] = { 0, 0, 0 };
		ret = clEnqueueReadImage(
			OpenclGPUPlatform::getOpenclPlatform().getCommandQueue(),
			output,
			CL_TRUE,
			origin,
			globalOutput,
			0,
			0,
			imgBuffer, 1, &sampleEvent, NULL);
		checkfatalError("read output fail", ret);
		std::cout << "\b\b\b\b" << ((std::chrono::duration<float>)(std::chrono::high_resolution_clock::now() - t0)).count() << " seconds" << std::endl;
		write2File(imgBuffer, filename, width, height);
	}

	//only for test use
	void test_init() {
		cl_int ret;
		superSampleformat.image_channel_data_type = CL_UNSIGNED_INT8;
		superSampleformat.image_channel_order = CL_RGBA;
		set2DDesc(superSampleDesc, CL_MEM_OBJECT_IMAGE2D, 640 * 4, 480 * 4);
		superSample = clCreateImage(
			OpenclGPUPlatform::getOpenclPlatform().getContext(),
			CL_MEM_READ_WRITE,
			&superSampleformat,
			&superSampleDesc,
			NULL,
			&ret);
		checkfatalError("Create super sample image buffer fail", ret);

		set2Dformat(outputformat, CL_RGBA, CL_UNSIGNED_INT8);
		set2DDesc(outputDesc, CL_MEM_OBJECT_IMAGE2D, 640, 480);
		output = clCreateImage(
			OpenclGPUPlatform::getOpenclPlatform().getContext(),
			CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
			&outputformat,
			&outputDesc,
			NULL,
			&ret);
		checkfatalError("Create output image buffer fail", ret);

		void* params[2];
		params[0] = (void *)&superSample;
		params[1] = (void *)&output;

		skym->genParameter(480 * 4, 640 * 4);
		samplem->genParameter(480, 640);
		skym->setParamters(params, 1);
		samplem->setParamters(params, 2);
	
	}
	void test_draw() {
		size_t globalsky[2] = {640*4, 480*4};//width first
		cl_int ret;
		cl_event skydrawEvent;
		ret = clEnqueueNDRangeKernel(
			OpenclGPUPlatform::getOpenclPlatform().getCommandQueue(),
			sskyk->getKernel(),
			2,
			NULL,
			globalsky, 
			skym->getOptSize2D(), 0, NULL, &skydrawEvent);
		checkfatalError("sky draw fail", ret);
		size_t globalsample[2] = { 640,  480};//width first
		cl_event sampleEvent;
		ret = clEnqueueNDRangeKernel(
			OpenclGPUPlatform::getOpenclPlatform().getCommandQueue(),
			ssamplek->getKernel(),
			2,
			NULL,
			globalsample,
			samplem->getOptSize2D(), 1, &skydrawEvent, &sampleEvent);
		checkfatalError("sample fail", ret);
		uint8_t *img = new uint8_t[640 * 480 * 4 * sizeof(uint8_t)];
		size_t origin[3] = { 0, 0, 0 }, globalregion[3] = {640, 480, 1};
		ret = clEnqueueReadImage(
			OpenclGPUPlatform::getOpenclPlatform().getCommandQueue(), 
			output, 
			CL_TRUE, 
			origin,
			globalregion,
			0,
			0,
			img, 1, &sampleEvent, NULL);
		checkfatalError("read output fail", ret);
		write2File(img, "testout.ppm", 640, 480);
		free(img);
	}
private:
	void write2File(uint8_t img[], char filename[], size_t width, size_t height) {
		std::ofstream ofs(filename, std::ios::out | std::ios::binary);
		ofs << "P6\n" << width << " " << height << "\n255\n";
		for (int i = 0; i < width*height*4;i++) {
			ofs << img[i];
			i++;
			ofs << img[i];
			i++;
			ofs << img[i];
			i++;
		}
		ofs.close();
	}
	void set2Dformat(cl_image_format &imFormat, cl_channel_order channelOrder, cl_channel_type channelType) {
		imFormat.image_channel_order = channelOrder;
		imFormat.image_channel_data_type = channelType;
	}
	void set2DDesc(cl_image_desc &desc, cl_mem_object_type type, size_t width, size_t height) {
		desc.image_type = type;
		desc.image_width = width;
		desc.image_height = height;
		//desc.image_depth = 1;
		//desc.image_array_size = 1;
		desc.image_row_pitch = 0;
		desc.image_slice_pitch = 0;
		desc.num_mip_levels = 0;
		desc.num_samples = 0;
		desc.buffer = NULL;
	}

	cl_mem lightTable;
	cl_mem sunDir;
	cl_mem sampleData;
	cl_mem sampleDataWH;
	cl_mem superSample;
	cl_mem output;
	
	float* table;
	size_t tableH;
	size_t tableW;
	cl_image_format tableformat;
	cl_image_desc tableDesc;

	size_t globalSample[2];
	size_t superSampleH;
	size_t superSampleW;
	cl_image_format superSampleformat;
	cl_image_desc superSampleDesc;

	cl_image_format outputformat;
	cl_image_desc outputDesc;

	size_t globalOutput[3];
	size_t width;
	size_t height;
	unsigned char *imgBuffer;

	float *samples;
	size_t sampleW;
	size_t sampleH;
	size_t samplenum;

	cl_kernel skyk;
	cl_kernel samplek;
	SimpleKernel *sskyk;
	SimpleKernel *ssamplek;
	OpenCLGPUManager *skym;
	OpenCLGPUManager *samplem;

	Atmosphere at;
	float sunDirV[3];
	float angle;
};
//column first
//generate as:
//x1,y1,w,x2,y2,w,x3,y3,w
float* genRectSample(unsigned w, unsigned h) {
	std::default_random_engine generator;
	generator.seed();
	std::uniform_real_distribution<float> distribution(0, 1); // to generate random floats in the range [0:1]

	unsigned totalSample = w*h;
	float weight = 1.f / (float)totalSample;
	float* randomData = new float[totalSample * 3];
	for (unsigned n = 0; n < h; ++n) {
		for (unsigned m = 0; m < w * 3; m+=3) {
			randomData[n*w * 3 + m] = ((float)m/3.f + distribution(generator))/w;
			randomData[n*w * 3 + m + 1] = ((float)n + distribution(generator))/h;
			randomData[n*w * 3 + m + 2] = weight;
		}
	}

	return randomData;
}

int main() {
#if 0
	float testSample[3] = { 0.f, 0.f, 1.f };
	SkyDrawer sk("nisSkyCL.cl", "skyFilter.cl", 640, 480, testSample, 1, 1, true);
#else
	unsigned samplenum = 4;
	float* sample = genRectSample(samplenum, samplenum);
	SkyDrawer sk("nisSkyCL.cl", "skyFilter.cl", 640, 480, sample, samplenum, samplenum, true);
#endif
	sk.init();
#if 1
	unsigned nangles = 128;
	for (unsigned i = 0; i < nangles; ++i) {
		char filename[1024];
		sprintf_s(filename, "./lookup/skydome.%04d.ppm", i);
		float angle = i / float(nangles - 1) * M_PI * 0.6;
		fprintf(stderr, "Rendering image %d, angle = %0.2f\n", i, angle * 180 / M_PI);
		sk.draw(Vec3f(0, cos(angle), -sin(angle)), filename);
	}
#else
	float angle = M_PI * 108.f / 127.f * 0.6;
	Vec3f sunDir(0, std::cos(angle), -std::sin(angle));
	sk.draw(sunDir);
#endif
	getchar();
	//sk.test_init();
	//sk.test_draw();
}

/*
int main() {
	cl_platform_id *platform_id = new cl_platform_id[2];
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;
	char show[1024];
	size_t renum;

	////////////////////////////initialize
	ret = clGetPlatformIDs(2, platform_id, &ret_num_platforms);
	std::cout << "platform num: " << ret_num_platforms << std::endl;
	ret = clGetPlatformInfo(platform_id[0], CL_PLATFORM_NAME, 1024, show, &renum);
	std::cout << "platform 1 " << show << std::endl;
	ret = clGetPlatformInfo(platform_id[1], CL_PLATFORM_NAME, 1024, show, &renum);
	std::cout << "platform 2 " << show << std::endl;

	std::cout << "use platform 1" << std::endl;
	ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
	std::cout << "GPU device_num " << ret_num_devices << std::endl;
	ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, 1024, show, &renum);
	std::cout << "Device 1 " << show <<std::endl;
	cl_uint unit;
	ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, 1024, &unit, &renum);
	std::cout << "unit num : " << unit << std::endl;
	cl_uint dim;
	ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, 1024, &dim, &renum);
	std::cout << "demonsion num " << dim << std::endl;
	size_t workitems[3];
	ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, 1024, workitems, &renum);
	std::cout << "demonsion ";
	for (int i = 0; i < 3; i++) {
		std::cout << workitems[i] << ", ";
	}
	std::cout << std::endl;
	size_t perUnit;
	ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, 1024, &perUnit, &renum);
	std::cout << "per Unit " << perUnit << std::endl;

	cl_ulong localBufferSize;
	ret = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, 1024, &localBufferSize, &renum);
	std::cout << "local memory size " << localBufferSize / (1024) <<"kb"<<std::endl;

	cl_ulong constBufferSize;
	ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, 1024, &constBufferSize, &renum);
	std::cout << "constant memory size " << constBufferSize/(1024) << "kb" << std::endl;

	cl_ulong globalBufferSize;
	ret = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, 1024, &globalBufferSize, &renum);
	std::cout << "global memory size " << globalBufferSize / (1024*1024) << "mb" << std::endl;

	std::cout << "build context with device 1 " << std::endl;
	cl_context context = NULL;
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	/////////////////////////////program

	cl_command_queue command_queue = NULL;
	std::cout << "build command queue with profiling" << std::endl;
	command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

	std::cout << "create buffer" << std::endl;
	cl_mem memobj = NULL;
	memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(char), NULL, &ret);

	cl_program program = NULL;
	char *source_str;
	size_t source_size;
	char fileName[] = "./hello.cl";
	FILE *fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	std::cout << "create program..." << std::endl;
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	
	std::cout << "complie..." << std::endl;
	char compileOpt[] = "-cl-no-signed-zeros";
	ret = clBuildProgram(program, 1, &device_id, compileOpt, NULL, NULL);
	if (ret != CL_SUCCESS) {
		std::cout << "build prgram fail " << ret << std::endl;
		int i;
		std::cin >> i;
		exit(0);
	}
	std::cout << "build sucess" << std::endl;

	////////////////////////kernel
	cl_kernel kernel = NULL;
	kernel = clCreateKernel(program, "hello", &ret);
	if (ret != CL_SUCCESS) {
		std::cout << "build kernel fail " << ret << std::endl;
		int i;
		std::cin >> i;
		exit(0);
	}
	size_t kernelGroupSize;
	ret = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, 1024, &kernelGroupSize, NULL);
	std::cout << "kernel group size " << kernelGroupSize << std::endl;

	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj);
	
	ret = clEnqueueTask(command_queue, kernel, 0, NULL, NULL);

	std::cout << "read result in block mode with no wait event" << std::endl;
	ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0, MEM_SIZE * sizeof(char), show, 0, NULL, NULL);
	puts(show);

	int i;
	std::cin >> i;
	delete[] platform_id;
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(memobj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	return 0;
}*/