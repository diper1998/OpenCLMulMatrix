#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 
#include <iostream> 
#include <CL/cl.hpp> 
#include <time.h>
#include <string>
#include <stdlib.h>
#include <windows.h>
#include "clinst.h"


int main(int argc, char* argv[]) {

	int* ptrSize = new int;
	int size = 0;
    int sizeCPU = 0;
    int sizeGPU = 0;
	int selectDevice = 0;
	bool flag = true;

	
	std::string sizeStr;
	std::string defDevice;
	std::string selectOptimization;


	for (int i = 0; i < 4; i++)
		if (argv[i] == NULL)
			exit(1);

	sizeStr = argv[1];
	*ptrSize = atoi(sizeStr.c_str());
	size = *ptrSize;

	defDevice = argv[2];

	selectOptimization = argv[3];

	float* A = new float[size * size];
	float* B = new float[size * size];
	float* C = new float[size * size];
	
	int numbArg = 4;
	int * sizeBuf = new int[numbArg];
	
	int * sizeBufMulOpt = new int[numbArg];

	for (int i = 0; i < size * size; i++) {
		A[i] = 1.0;
		B[i] = 2.0;
		C[i] = 3.0;
	}

	sizeBuf[0] = sizeof(float)*size*size;
	sizeBuf[1] = sizeof(float)*size*size;
	sizeBuf[2] = sizeof(float)*size*size;
	sizeBuf[3] = sizeof(int*);



	clmulmatrix mulMatrix(defDevice, numbArg, sizeBuf, A, B, C, ptrSize);
	clmulmatrixmulopt mulMatrixOpt(A, B, C, *ptrSize, numbArg, sizeBuf);

    

	if (selectOptimization == "opt") {

		//mulMatrix.GetInfo();
          std::cout << "SIZE: " << *ptrSize << std::endl;
          std::cout << "CPU: "<< *ptrSize << " GPU: 0"  << std::endl;

	  mulMatrix.MulMatrixOpt();
	 

	}
	else {
		if (selectOptimization == "mulopt") {

            

            sizeStr = argv[4];
            sizeCPU = atoi(sizeStr.c_str());
            sizeStr = argv[5];
            sizeGPU = atoi(sizeStr.c_str());
            
			mulMatrixOpt.sizeCPU = sizeCPU;
			mulMatrixOpt.sizeGPU = sizeGPU;

            std::cout << "SIZE: " << sizeCPU + sizeGPU << std::endl;
                        std::cout << "CPU: " << sizeCPU << " GPU: " << sizeGPU
                                  << std::endl;

			//mulMatrixOpt.GetInfo();
			mulMatrixOpt.MulMatrixMulOpt();
			
		}
		else {
			//mulMatrix.GetInfo();
             std::cout << "SIZE: " << *ptrSize << std::endl;
             std::cout << "GPU: \n";
		     mulMatrix.MulMatrix();

		}
	}

	


	delete[]A;
	delete[]B;
	delete[]C;

	return 0;



}


