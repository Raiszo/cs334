#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__
void global_reduce(float *d_out, float *d_in)
{
	const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
	
}

int main(int argc, char **argv)
{
	int deviceCount;

	const int ARRAY_SIZE = 13;
	const int ARRAY_BYTES = ARRAY_SIZE * ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
	float h_in[ARRAY_SIZE][ARRAY_SIZE];
	float sum = 0.0f;
	for(int i = 0; i < ARRAY_SIZE; i++) {
		for (int j = 0; j < ARRAY_SIZE; j++) {
			// generate random float in [-1.0f, 1.0f]
			h_in[i][j] = -1.0f + (float)rand()/((float)RAND_MAX/2.0f);
			sum += h_in[i][j];
		}
	}

	// declare GPU memory pointers
	float * d_in, * d_intermediate, * d_out;

	// allocate GPU memory
	cudaMalloc((void **) &d_in, ARRAY_BYTES);
	//cudaMalloc((void **) &d_intermediate, ARRAY_BYTES); // overallocated
	cudaMalloc((void **) &d_out, sizeof(float));

	// transfer the input array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 


	// copy back the sum from GPU
	float h_out;
	cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

	printf("average time elapsed: %f\n", elapsedTime);

	// free GPU memory allocation
	cudaFree(d_in);
	cudaFree(d_intermediate);
	cudaFree(d_out);
        
	return 0;
}
