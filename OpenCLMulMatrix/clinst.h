#pragma once
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 
#include <iostream> 
#include <CL/cl.hpp> 
#include <time.h>
#include <string>
#include <stdlib.h>
#include <windows.h>

class clinst
{
public:

	std::vector<cl::Platform> allPlatforms;    
	cl::Platform defaultPlatform;
	cl::Platform defaultPlatformCPU;
	cl::Platform defaultPlatformGPU;

	std::vector<cl::Platform> allPlatformCPU;
	std::vector<cl::Platform> allPlatformGPU;


	std::vector<cl::Device> allDevices;
	cl::Device defaultDevice;

	std::vector<cl::Device> allDeviceCPU;
	std::vector<cl::Device> allDeviceGPU;

	cl::Device defaultDeviceCPU;
	cl::Device defaultDeviceGPU;

	cl::Context myContext;

	cl::Program::Sources mySources;
	
	std::string kernelCode;
	int numbArguments;

	cl::Program myProgram;

	cl::CommandQueue commandQueue;

	cl::Buffer *myBuffers;

	int numbArg;
	int* bufferOfSizes;

	clinst(std::string defDevice);
	void SetKernelCode(std::string kerCode);
	
	~clinst();

	int GetInfo();
};

class clmulmatrix : public clinst {

public:

	float* A;
	float* B;
	float* C;
	int size;
	const cl_int BLOCK_SIZE = 16;

	clmulmatrix(std::string defDevice, int numbArg, int* sizeBuf, float* arrayA, float* arrayB, float* arrayC, int* sizeArray);
    
	~clmulmatrix();

	void MulMatrix();
	void MulMatrixOpt();

	void ShowMatrix();
};

class clmulmatrixmulopt {
	
public:

	std::vector<cl::Platform> allPlatforms;
	cl::Platform defaultPlatform;
	cl::Platform defaultPlatformCPU;
	cl::Platform defaultPlatformGPU;

	std::vector<cl::Platform> allPlatformCPU;
	std::vector<cl::Platform> allPlatformGPU;


	//std::vector<cl::Device> allDevices;
	//cl::Device defaultDevice;

	std::vector<cl::Device> allDeviceCPU;
	std::vector<cl::Device> allDeviceGPU;

	cl::Device defaultDeviceCPU;
	cl::Device defaultDeviceGPU;

	cl::Context myContext;

	cl::Program::Sources mySources;

	std::string kernelCodeCPU;
	std::string kernelCodeGPU;
	

	cl::Program myProgramCPU;

	cl::Program myProgramGPU;

	cl::CommandQueue commandQueueCPU;

	cl::CommandQueue commandQueueGPU;

	cl::Buffer *myBuffers;
	int numbArg;
	int* bufferOfSizes;

	float* A;
	float* B;
	float* C;
	int size;

	int sizeGPU;
	int sizeCPU;

	clmulmatrix* ptrMulCPU;
	clmulmatrix* ptrMulGPU;

	const cl_int BLOCK_SIZE = 16;

    clmulmatrixmulopt(float* matrixA, float* matrixB, float* matrixC, int sizeMatrix, int numberArg, int* bufferSizes);

	int GetInfo();
	void MulMatrixMulOpt();

};