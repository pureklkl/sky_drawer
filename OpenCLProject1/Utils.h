#pragma once
#include<iostream>
#include<string>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

void fatalError(char msg[]) {
	std::cout << "Error: " << msg <<std::endl;
	getchar();
	std::exit(0);
}
void checkfatalError(char msg[], cl_int ret) {
	if (ret != CL_SUCCESS) {
		std::cout << "Error code:" << ret << std::endl;
		fatalError(msg);
	}
}
void lineMsg(const char msg[]) {
	std::cout << "Info: " << msg << std::endl;
}
std::string byte2Kb(cl_ulong bs) {
	return (std::to_string(bs >> 10)+"kb");
}
std::string byte2Mb(cl_ulong bs) {
	return (std::to_string(bs >> 20) + "Mb");
}
