#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__
void reduce_max(float *d_in,
								const int numRows, const int numCols,
								float *d_out)
{
	const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
	const int totalThreads = gridDim.x * blockDim.x * gridDim.y * blockDim.y;
	const int index = thread_2D_pos.y * numCols + thread_2D_pos.x;
	const int absIndex = thread_2D_pos.y * gridDim.x*blockDim.x + thread_2D_pos.x;

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		d_in[absIndex] = -1.0f;
		
	for (unsigned int s = totalThreads / 2; s > 0; s >>= 1) {
		if (absIndex < s)
			d_in[absIndex] = max(d_in[absIndex], d_in[absIndex + s]);
		
		__syncthreads();
			    
	}

	// only thread 0 writes result for this block back to global mem
	if (absIndex == 0) {
		d_out[absIndex] = d_in[absIndex];
	}
}

// Need numRows and numCols to pad to min value
__global__
void shared_reduce_max(float *d_in,
											 const int numRows, const int numCols,
											 float *d_out,
											 )
{
	extern __shared__ float simg[];
	const int2 image_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
																			 blockIdx.y * blockDim.y + threadIdx.y);
	const int index = image_2D_pos.x + numCols * image_2D_pos.y;
	const int bid = threadIdx.x + blockDim.x * threadIdx.y;
	
	if (image_2D_pos.x >= numCols || image_2D_pos.y >= numRows)
		simg[bid] = -1.0f;
	else
		simg[bid] = d_in[index];
	
	__syncthreads();

	for (unsigned int s = blockDim.x*blockDim.y / 2; s > 0; s >>= 1) {
		if (bid < s)
			simg[bid] = max(simg[bid], simg[bid + s]);
		
		__syncthreads();
	}

	if (bid == 0) {
		d_out[blockIdx.x + gridDim.x * blockIdx.y] = simg[0];
	}
}

int main(int argc, char **argv)
{
	// TODO, use dynamic array :'v
	const int ARRAY_SIZE = 13;
	const int ARRAY_BYTES = ARRAY_SIZE * ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
	float h_in[ARRAY_SIZE][ARRAY_SIZE];
	float maxi = -1.0f;
	
	for(int i = 0; i < ARRAY_SIZE; i++) {
		for (int j = 0; j < ARRAY_SIZE; j++) {
			// generate random float in [-1.0f, 1.0f]
			h_in[i][j] = -1.0f + (float)rand()/((float)RAND_MAX/2.0f);
			maxi = max(maxi, h_in[i][j]);
		}
	}
	printf("Expected max: %f", maxi);
	printf("\n");
	
	for (int i =0; i < ARRAY_SIZE; i++) {
		for (int j = 0; j < ARRAY_SIZE; j++) {
			printf("%f", h_in[i][j]);
			printf("  ");
		}
		printf("\n");
	}

	// guess image is no bigger than 1024*1024
	// declare GPU memory pointers
	float * d_in, * d_inter, * d_out;
	// Grid and Block size
	const int size = 4;
	const dim3 blockSize(size, size, 1);
	const dim3 gridSize((int) ARRAY_SIZE/size + 1, (int) ARRAY_SIZE/size + 1, 1);

	int shared_size = sizeof(float) * size * size;

	// allocate GPU memory
	cudaMalloc((void **) &d_in, ARRAY_BYTES);
	cudaMalloc((void **) &d_inter, (ARRAY_SIZE/size+1) * (ARRAY_SIZE/size+1) * sizeof(float));
	cudaMalloc((void **) &d_out, sizeof(float));

	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 


	shared_reduce_max<<< gridSize, blockSize, shared_size >>>(d_in,
																														ARRAY_SIZE, ARRAY_SIZE,
																														d_inter);
	cudaDeviceSynchronize();
	
	shared_reduce_max<<< (1,1,1), gridSize ,sizeof(float) >>>(d_inter,
																														ARRAY_SIZE/size+1, ARRAY_SIZE/size+1,
																														d_out);
	cudaDeviceSynchronize();

	// copy back the sum from GPU
	float h_out;
	cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

	printf("%f", h_out);
	printf("\n");
	// free GPU memory allocation
	cudaFree(d_in);
	cudaFree(d_inter);
	cudaFree(d_out);
        
	return 0;
}
