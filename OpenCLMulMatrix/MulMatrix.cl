	void kernel MulMatrix(global const float* A, global const float* B, global float* C, constant int* size_ptr){ 
    	int i = get_global_id(0);																				  
		int j = get_global_id(1);																				  
		int size = *size_ptr;																					  
	   for( int k = 0; k < size; k++){																			  
			C[i*size+j]+=A[i*size+k]*B[k*size+j];																  
       }																										  	
	 }																							                   

	 void kernel MulMatrixOpt( __global float * a, __global float * b, __global float * c, constant int* size_ptr, __local float * a_local,__local float * b_local, int BLOCK_SIZE){"																				  																			  
		int i = get_global_id(0);														  
		int j = get_global_id(1);														  
		int size = *size_ptr;															  
		int localI = get_local_id(0);													  
		int localJ = get_local_id(1);													  
		float sum = 0.0f;																  
		for (int p = 0; p < size / BLOCK_SIZE; ++p)	{						      
			a_local[localI * BLOCK_SIZE + localJ] =	a[i * size + p * BLOCK_SIZE + localJ];   
			b_local[localI * BLOCK_SIZE + localJ] = b[(p * BLOCK_SIZE + localI) * size + j]; 
			barrier(CLK_LOCAL_MEM_FENCE);												  
			for (int l = 0; l < BLOCK_SIZE; ++l){										  
				sum += a_local[localI * BLOCK_SIZE + l] * b_local[l * BLOCK_SIZE + localJ]; 
         }
			barrier(CLK_LOCAL_MEM_FENCE);												  
		}																		          
		c[i * size + j] = sum; 															  
	    };