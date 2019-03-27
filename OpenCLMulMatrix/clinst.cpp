#include "clinst.h"

clinst::clinst(std::string defDevice = "CPU") {
  cl::Platform::get(&allPlatforms);

  if (allPlatforms.size() == 0) {
    std::cout << "Platforms are not found. Check OpenCL installation!\n";
  }

  for (int i = 0; i < allPlatforms.size(); i++) {
    allPlatforms[i].getDevices(CL_DEVICE_TYPE_CPU, &allDeviceCPU);
    if (allDeviceCPU.size() != 0) {
      defaultDeviceCPU = allDeviceCPU[0];
      defaultPlatformCPU = allPlatforms[i];
      break;
    }
  }

  for (int i = 0; i < allPlatforms.size(); i++) {
    allPlatforms[i].getDevices(CL_DEVICE_TYPE_GPU, &allDeviceGPU);

    if (allDeviceGPU.size() != 0) {
      defaultDeviceGPU = allDeviceGPU[0];
      defaultPlatformGPU = allPlatforms[i];
      break;
    }
  }

  // int selectDevice;
  if (defDevice == "CPU") {
    defaultPlatform = defaultPlatformCPU;
    defaultDevice = defaultDeviceCPU;
  } else if (defDevice == "GPU") {
    defaultPlatform = defaultPlatformGPU;
    defaultDevice = defaultDeviceGPU;
  } else {
    std::cout << " Your device is not found. Try CPU or GPU!\n";
    exit(1);
  };

  // defaultPlatform = allPlatforms[0];

  // defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);

  /*if (allDevices.size() == 0) {
          std::cout << "Devices are not found. Check OpenCL installation!\n";
          exit(1);
  }*/

  /*
  int selectDevice;
  if (defDevice == "CPU") selectDevice = 1;
  else if (defDevice == "GPU") selectDevice = 0;
  else {
          std::cout << " Your device is not found. Try CPU or GPU!\n";
                  exit(1);
  };
  

   defaultDevice = allDevices[selectDevice];
  */

  cl::Context makeContext({defaultDevice});
  myContext = makeContext;
}

void clinst::SetKernelCode(std::string kerCode) {
  kernelCode = kerCode;
  mySources.push_back({kernelCode.c_str(), kernelCode.length()});

  cl::Program makeProgram(myContext, mySources);
  myProgram = makeProgram;

  if (myProgram.build({defaultDevice}) != CL_SUCCESS) {
    std::cout << " Error building: "
              << myProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDevice)
              << "\n";
    exit(1);
  }

  myBuffers = new cl::Buffer[numbArg];

  for (int i = 0; i < numbArg; i++) {
    cl::Buffer buffer(myContext, CL_MEM_READ_WRITE, bufferOfSizes[i]);
    myBuffers[i] = buffer;
  }

  cl::CommandQueue comQueue(myContext, defaultDevice);
  commandQueue = comQueue;
}

clinst::~clinst() {}

int clinst::GetInfo() {
  std::string myInfo;

  myInfo = defaultPlatform.getInfo<CL_PLATFORM_NAME>();
  std::cout << std::endl << myInfo << std::endl;
  myInfo = defaultDevice.getInfo<CL_DEVICE_NAME>();
  std::cout << std::endl << myInfo << std::endl;

  return 0;
}

clmulmatrix::clmulmatrix(std::string defDevice, int numberArg, int* bufferSizes,
                         float* matrixA, float* matrixB, float* matrixC,
                         int* sizeMatrix)
    : clinst(defDevice) {
  size = *sizeMatrix;
  numbArg = numberArg;
  bufferOfSizes = new int[numbArg];

  for (int i = 0; i < numbArg; i++) {
    bufferOfSizes[i] = bufferSizes[i];
  }

  A = new float[size * size];
  B = new float[size * size];
  C = new float[size * size];

  for (int i = 0; i < size * size; i++) {
    A[i] = matrixA[i];
    B[i] = matrixB[i];
    C[i] = matrixC[i];
  }
}


clmulmatrix::~clmulmatrix() {}

void clmulmatrix ::MulMatrix() {
  cl_int error;
  bool flag = true;
  LARGE_INTEGER startTime, endTime, freq;

  kernelCode =
      "void kernel MulMatrix(global const float* A, global const float* B, "
      "global float* C, constant int* size_ptr){ "
      " int j = get_global_id(0);"
      " int i = get_global_id(1);  "
      " int size = *size_ptr;"
      "for( int k = 0; k < size; k++){ "
      "C[i*size+j]+=A[i*size+k]*B[k*size+j]; "
      " 			 } }";

  SetKernelCode(kernelCode);

  error = commandQueue.enqueueWriteBuffer(myBuffers[0], CL_TRUE, 0,
                                          sizeof(float) * size * size, A);
  if (error != CL_SUCCESS) {
    std::cout << "Error writing\n" << error;
  }
  error = commandQueue.enqueueWriteBuffer(myBuffers[1], CL_TRUE, 0,
                                          sizeof(float) * size * size, B);
  if (error != CL_SUCCESS) {
    std::cout << "Error writing\n";
  }
  error = commandQueue.enqueueWriteBuffer(myBuffers[3], CL_TRUE, 0,
                                          sizeof(int*), &size);
  if (error != CL_SUCCESS) {
    std::cout << "Error writing\n";
  }

  // run the kernel
  cl::Kernel mulMatrix(myProgram, "MulMatrix");
  error = mulMatrix.setArg(0, myBuffers[0]);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg A\n";
  }
  error = mulMatrix.setArg(1, myBuffers[1]);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg B\n";
  }
  error = mulMatrix.setArg(2, myBuffers[2]);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg C\n";
  }
  error = mulMatrix.setArg(3, myBuffers[3]);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg size\n";
  }

  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&startTime);

  error = commandQueue.enqueueNDRangeKernel(
      mulMatrix, cl::NullRange, cl::NDRange(size, size), cl::NullRange);

  commandQueue.finish();

  QueryPerformanceCounter(&endTime);

  double my_time =
      (endTime.QuadPart - startTime.QuadPart) / (double)freq.QuadPart;

  if (error != CL_SUCCESS) {
    std::cout << "Error queue";
  }

  // read result C from the device to array C
  commandQueue.enqueueReadBuffer(myBuffers[2], CL_TRUE, 0,
                                 sizeof(float) * size * size, C);

  if (error != CL_SUCCESS) {
    std::cout << "Error reading\n";
  }

  for (int i = 0; i < size * size; i++) {
    if (C[i] != size * 2) {
      std::cout << "Test is failed\n";
      flag = false;
      break;
    }
  }

  if (flag) {
    std::cout << "Test is successful\n";
    std::cout << "Time: " << my_time << std::endl;
  }
};

void clmulmatrix ::MulMatrixOpt() {
  cl_int error;
  bool flag = true;
  LARGE_INTEGER startTime, endTime, freq;

  kernelCode =
      "	void kernel MulMatrixOpt( __global float * a, __global float * b, "
      "__global float * c, constant int* size_ptr, __local float * "
      "a_local,__local float * b_local, int BLOCK_SIZE){"
      "int j = get_global_id(0);					"
      "									  "
      "int i = get_global_id(1);					"
      "									  "
      "int size = *size_ptr;						"
      "									  "
      "int localJ = get_local_id(0);					"
      "								  "
      "int localI = get_local_id(1);					"
      "								  "
      "float sum = 0.0f;						"
      "									"
      "	  "
      "for (int p = 0; p < size / BLOCK_SIZE; ++p)	{		"
      "				      "
      "	a_local[localI * BLOCK_SIZE + localJ] =	a[i * size + p * BLOCK_SIZE + "
      "localJ];   "
      "	b_local[localI * BLOCK_SIZE + localJ] = b[(p * BLOCK_SIZE + localI) * "
      "size + j]; "
      "	barrier(CLK_LOCAL_MEM_FENCE);					"
      "							  "
      "	for (int l = 0; l < BLOCK_SIZE; ++l){				"
      "						  "
      "		sum += a_local[localI * BLOCK_SIZE + l] * b_local[l * "
      "BLOCK_SIZE + localJ]; "
      " }"
      "	barrier(CLK_LOCAL_MEM_FENCE);					"
      "							  "
      "}								"
      "									"
      "	          "
      "c[i * size + j] = sum; 						"
      "									  "
      "}";

  SetKernelCode(kernelCode);

  error = commandQueue.enqueueWriteBuffer(myBuffers[0], CL_TRUE, 0,
                                          sizeof(float) * size * size, A);
  if (error != CL_SUCCESS) {
    std::cout << "Error writing\n";
  }
  error = commandQueue.enqueueWriteBuffer(myBuffers[1], CL_TRUE, 0,
                                          sizeof(float) * size * size, B);
  if (error != CL_SUCCESS) {
    std::cout << "Error writing\n";
  }
  error = commandQueue.enqueueWriteBuffer(myBuffers[3], CL_TRUE, 0,
                                          sizeof(int*), &size);
  if (error != CL_SUCCESS) {
    std::cout << "Error writing\n";
  }

  // run the kernel
  cl::Kernel MulMatrix(myProgram, "MulMatrixOpt");

  error = MulMatrix.setArg(0, myBuffers[0]);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg A\n";
  }
  error = MulMatrix.setArg(1, myBuffers[1]);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg B\n";
  }
  error = MulMatrix.setArg(2, myBuffers[2]);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg C\n";
  }
  error = MulMatrix.setArg(3, myBuffers[3]);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg size\n";
  }

  error = MulMatrix.setArg(4, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, NULL);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg size\n";
  }

  error = MulMatrix.setArg(5, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, NULL);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg size\n";
  }

  int blockSize = BLOCK_SIZE;

  error = MulMatrix.setArg(6, sizeof(cl_int), &blockSize);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg size\n";
  }

  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&startTime);

  error = commandQueue.enqueueNDRangeKernel(
      MulMatrix, cl::NullRange, cl::NDRange(size, size),
      cl::NDRange(BLOCK_SIZE, BLOCK_SIZE));

  commandQueue.finish();
  QueryPerformanceCounter(&endTime);

  double my_time =
      (endTime.QuadPart - startTime.QuadPart) / (double)freq.QuadPart;

  if (error != CL_SUCCESS) {
    std::cout << "Error queue";
  }

  // read result C from the device to array C
  commandQueue.enqueueReadBuffer(myBuffers[2], CL_TRUE, 0,
                                 sizeof(float) * size * size, C);

  if (error != CL_SUCCESS) {
    std::cout << "Error reading\n";
  }

  for (int i = 0; i < size * size; i++) {
    if (C[i] != size * 2) {
      std::cout << "Test is failed\n";
      flag = false;
      break;
    }
  }

  if (flag) {
    std::cout << "Test is successful\n";
    std::cout << "Time: " << my_time << std::endl;
  }
}

void clmulmatrix::ShowMatrix() {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) std::cout << C[i * size + j] << " ";
    std::cout << std::endl;
  }
}

clmulmatrixmulopt::clmulmatrixmulopt(float* matrixA, float* matrixB,
                                     float* matrixC, int sizeMatrix,
                                     int numberArg, int* bufferSizes) {
  size = sizeMatrix;
  numbArg = numberArg;
  bufferOfSizes = new int[numbArg];

  for (int i = 0; i < numbArg; i++) {
    bufferOfSizes[i] = bufferSizes[i];
  }

  A = new float[size * size];
  B = new float[size * size];
  C = new float[size * size];

  for (int i = 0; i < size * size; i++) {
    A[i] = matrixA[i];
    B[i] = matrixB[i];
    C[i] = 0;
  }

  cl::Platform::get(&allPlatforms);
  
  

  for (int i = 0; i < allPlatforms.size(); i++) {
    // allPlatforms[i].getDevices(CL_DEVICE_TYPE_CPU, &allDeviceCPU);
    allPlatforms[i].getDevices(CL_DEVICE_TYPE_GPU, &allDeviceGPU);
    allPlatforms[i].getDevices(CL_DEVICE_TYPE_CPU, &allDeviceCPU);

    if (allDeviceGPU.size() != 0 && allDeviceCPU.size() != 0) {
      defaultDeviceGPU = allDeviceGPU[0];
      defaultDeviceCPU = allDeviceCPU[0];
      defaultPlatformGPU = allPlatforms[i];
      defaultPlatformCPU = allPlatforms[i];
      break;
    }
  }

  
  /*
  clmulmatrix mulCPU_((std::string)"CPU", numbArg, bufferOfSizes, A, B, C,
  &size); clmulmatrix mulGPU_((std::string)"GPU", numbArg, bufferOfSizes, A, B,
  C, &size);

  ptrMulCPU = &mulCPU_;
  ptrMulGPU = &mulGPU_;
*/
}

int clmulmatrixmulopt::GetInfo() {
  std::string myInfo;

  std::cout << std::endl << "CPU: " << std::endl;
  myInfo = defaultPlatformCPU.getInfo<CL_PLATFORM_NAME>();
  std::cout << std::endl << myInfo << std::endl;
  myInfo = defaultDeviceCPU.getInfo<CL_DEVICE_NAME>();
  std::cout << std::endl << myInfo << std::endl;

  std::cout << std::endl << "GPU: " << std::endl;
  myInfo = defaultPlatformGPU.getInfo<CL_PLATFORM_NAME>();
  std::cout << std::endl << myInfo << std::endl;
  myInfo = defaultDeviceGPU.getInfo<CL_DEVICE_NAME>();
  std::cout << std::endl << myInfo << std::endl;

  return 0;
}

void clmulmatrixmulopt::MulMatrixMulOpt() {
  cl_int error;
  bool flag = true;
  LARGE_INTEGER startTime, endTime, freq;

  cl::Context makeContext({defaultDeviceCPU, defaultDeviceGPU});

  myContext = makeContext;

  kernelCodeCPU =
      "	void kernel MulMatrixOptCPU( __global float * a, __global float * b, "
      "__global float * c, constant int* size_ptr, __local float * "
      "a_local,__local float * b_local, int BLOCK_SIZE){"
      "int j = get_global_id(0);					"
      "									  "
      "int i = get_global_id(1);					"
      "									  "
      "int size = *size_ptr;						"
      "									  "
      "int localJ = get_local_id(0);					"
      "								  "
      "int localI = get_local_id(1);					"
      "								  "
      "float sum = 0.0f;						"
      "									"
      "	  "
      "for (int p = 0; p < size / BLOCK_SIZE; ++p)	{		"
      "				      "
      "	a_local[localI * BLOCK_SIZE + localJ] =	a[i * size + p * BLOCK_SIZE + "
      "localJ];   "
      "	b_local[localI * BLOCK_SIZE + localJ] = b[(p * BLOCK_SIZE + localI) * "
      "size + j]; "
      "	barrier(CLK_LOCAL_MEM_FENCE);					"
      "							  "
      "	for (int l = 0; l < BLOCK_SIZE; ++l){				"
      "						  "
      "		sum += a_local[localI * BLOCK_SIZE + l] * b_local[l * "
      "BLOCK_SIZE + localJ]; "
      " }"
      "	barrier(CLK_LOCAL_MEM_FENCE);					"
      "							  "
      "}								"
      "									"
      "	          "
      "c[i * size + j] = sum; 						"
      "									  "
      "}"
      "	void kernel MulMatrixOptGPU( __global float * a, __global float * b, "
      "__global float * c, constant int* size_ptr, __local float * "
      "a_local,__local float * b_local, int BLOCK_SIZE, int offset){"
      "int j = get_global_id(0)+offset;				"
      "									"
      "	  "
      "int i = get_global_id(1);					"
      "									  "
      "int size = *size_ptr;						"
      "									  "
      "int localJ = get_local_id(0);					"
      "								  "
      "int localI = get_local_id(1);					"
      "								  "
      "float sum = 0.0f;"
      "for (int p = 0; p < size / BLOCK_SIZE; ++p)	{		"
      "				      "
      "	a_local[localI * BLOCK_SIZE + localJ] =	a[i * size + p * BLOCK_SIZE + "
      "localJ];   "
      "	b_local[localI * BLOCK_SIZE + localJ] = b[(p * BLOCK_SIZE + localI) * "
      "size + j]; "
      "	barrier(CLK_LOCAL_MEM_FENCE);					"
      "							  "
      "	for (int l = 0; l < BLOCK_SIZE; ++l){				"
      "						  "
      "		sum += a_local[localI * BLOCK_SIZE + l] * b_local[l * "
      "BLOCK_SIZE + localJ]; "
      " }"
      "	barrier(CLK_LOCAL_MEM_FENCE);					"
      "							  "
      "}"
      "c[i * size + j] = sum; 						"
      "									  "
      "}";

    mySources.push_back({kernelCodeCPU.c_str(), kernelCodeCPU.length()});

  cl::Program makeProgramCPU(myContext, mySources);
  myProgramCPU = makeProgramCPU;

  if (myProgramCPU.build({defaultDeviceCPU, defaultDeviceGPU}) != CL_SUCCESS) {
    std::cout
        << " Error building: "
        << myProgramCPU.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDeviceCPU)
        << "\n"
        << myProgramCPU.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDeviceGPU)
        << "\n";
    exit(1);
  } 

  myBuffers = new cl::Buffer[numbArg];

  for (int i = 0; i < numbArg; i++) {
    cl::Buffer buffer(myContext, CL_MEM_READ_WRITE, bufferOfSizes[i]);
    myBuffers[i] = buffer;
  }

  cl::CommandQueue comQueueCPU(myContext, defaultDeviceCPU);
  commandQueueCPU = comQueueCPU;

  cl::CommandQueue comQueueGPU(myContext, defaultDeviceGPU);
  commandQueueGPU = comQueueGPU;

  error = commandQueueCPU.enqueueWriteBuffer(myBuffers[0], CL_TRUE, 0,
                                             sizeof(float) * size * size, A);
  if (error != CL_SUCCESS) {
    std::cout << "Error writing CPU A\n";
  }
  error = commandQueueCPU.enqueueWriteBuffer(myBuffers[1], CL_TRUE, 0,
                                             sizeof(float) * size * size, B);
  if (error != CL_SUCCESS) {
    std::cout << "Error writing CPU B\n";
  }

  error = commandQueueCPU.enqueueWriteBuffer(myBuffers[2], CL_TRUE, 0,
                                             sizeof(float) * size * size, C);
  if (error != CL_SUCCESS) {
    std::cout << "Error writing CPU C\n";
  }

  error = commandQueueCPU.enqueueWriteBuffer(myBuffers[3], CL_TRUE, 0,
                                             sizeof(int*), &size);
  if (error != CL_SUCCESS) {
    std::cout << "Error writing CPU &size\n";
  }

  error = commandQueueGPU.enqueueWriteBuffer(myBuffers[0], CL_TRUE, 0,
                                             sizeof(float) * size * size, A);
  if (error != CL_SUCCESS) {
    std::cout << "Error writing GPU A\n";
  }
  error = commandQueueGPU.enqueueWriteBuffer(myBuffers[1], CL_TRUE, 0,
                                             sizeof(float) * size * size, B);
  if (error != CL_SUCCESS) {
    std::cout << "Error writing GPU B\n";
  }

  error = commandQueueGPU.enqueueWriteBuffer(myBuffers[2], CL_TRUE, 0,
                                             sizeof(float) * size * size, C);
  if (error != CL_SUCCESS) {
    std::cout << "Error writing GPU C\n";
  }

  error = commandQueueGPU.enqueueWriteBuffer(myBuffers[3], CL_TRUE, 0,
                                             sizeof(int*), &size);


  if (error != CL_SUCCESS) {
    std::cout << "Error writing &size\n";
  }

  // run the kernel
  cl::Kernel MulMatrixCPU(myProgramCPU, "MulMatrixOptCPU");
  cl::Kernel MulMatrixGPU(myProgramCPU, "MulMatrixOptGPU");

  error = MulMatrixCPU.setArg(0, myBuffers[0]);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg CPU A\n";
  }

  error = MulMatrixGPU.setArg(0, myBuffers[0]);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg GPU A\n";
  }

  error = MulMatrixCPU.setArg(1, myBuffers[1]);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg CPU B \n";
  }

  error = MulMatrixGPU.setArg(1, myBuffers[1]);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg GPU B\n";
  }

  error = MulMatrixCPU.setArg(2, myBuffers[2]);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg CPU C\n";
  }

  error = MulMatrixGPU.setArg(2, myBuffers[2]);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg GPU C\n";
  }

  error = MulMatrixCPU.setArg(3, myBuffers[3]);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg CPU size\n";
  }
  error = MulMatrixGPU.setArg(3, myBuffers[3]);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg GPU size\n";
  }

  error = MulMatrixCPU.setArg(4, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, NULL);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg size\n";
  }

  error = MulMatrixCPU.setArg(5, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, NULL);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg size\n";
  }

  int blockSize = BLOCK_SIZE;

  error = MulMatrixCPU.setArg(6, sizeof(cl_int), &blockSize);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg size\n";
  }

  error = MulMatrixGPU.setArg(4, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, NULL);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg size\n";
  }

  error = MulMatrixGPU.setArg(5, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, NULL);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg size\n";
  }

  error = MulMatrixGPU.setArg(6, sizeof(cl_int), &blockSize);
  if (error != CL_SUCCESS) {
    std::cout << "Error setArg size\n";
  }

  error = MulMatrixGPU.setArg(7, sizeof(cl_int), &sizeCPU);
  if (error != CL_SUCCESS) {
	  std::cout << "Error setArg size\n";
  }


  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&startTime);

  // cl::NDRange(mulCPU.BLOCK_SIZE, mulCPU.BLOCK_SIZE)
  // cl::NDRange(mulGPU.BLOCK_SIZE, mulGPU.BLOCK_SIZE)
  error = commandQueueCPU.enqueueNDRangeKernel(
      MulMatrixCPU, cl::NDRange(0, 0), cl::NDRange(sizeCPU, size),
      cl::NDRange(BLOCK_SIZE, BLOCK_SIZE));
  // error = commandQueueGPU.enqueueNDRangeKernel(MulMatrixGPU, cl::NDRange(size
  // / 2, 0), cl::NDRange(size, size), cl::NDRange(BLOCK_SIZE, BLOCK_SIZE));
  error = commandQueueGPU.enqueueNDRangeKernel(
      MulMatrixGPU, cl::NDRange(0, 0), cl::NDRange(sizeGPU, size),
      cl::NDRange(BLOCK_SIZE, BLOCK_SIZE));

  commandQueueGPU.finish();
  commandQueueCPU.finish();

  QueryPerformanceCounter(&endTime);

  double my_time =
      (endTime.QuadPart - startTime.QuadPart) / (double)freq.QuadPart;

  if (error != CL_SUCCESS) {
    std::cout << "Error queue";
  }

  // read result C from the device to array C
  // commandQueueCPU.enqueueReadBuffer(myBuffers[2], CL_TRUE, 0, (sizeof(float)
  // * size * size)/2, C); commandQueueCPU.finish(); commandQueueGPU.finish();

  commandQueueGPU.enqueueReadBuffer(myBuffers[2], CL_TRUE, 0,
                                    (sizeof(float) * size * size), C);

  if (error != CL_SUCCESS) {
    std::cout << "Error reading\n";
  }

  /*for (int i = 0; i < size; i++) {
          for (int j = 0; j < size; j++){

                  std::cout << C[i*size + j]<< " " ;

  }
          std::cout << std::endl;
}*/

  for (int i = 0; i < size * size; i++) {
    if (C[i] != size * 2) {
      std::cout << "Test is failed\n";
      flag = false;
      break;
    }
  }

  if (flag) {
    std::cout << std::endl << "Test is successful\n";
  }

  std::cout << "Time: " << my_time << std::endl;
}
