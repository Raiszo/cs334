#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Need numRows and numCols to pad to min value
__global__
void reduce_max(float *d_in,
											 const int numRows, const int numCols,
											 float *d_out)
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

__global__
void reduce_min(float *d_in,
											 const int numRows, const int numCols,
											 float *d_out)
{
	extern __shared__ float simg[];
	const int2 image_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
																			 blockIdx.y * blockDim.y + threadIdx.y);
	const int index = image_2D_pos.x + numCols * image_2D_pos.y;
	const int bid = threadIdx.x + blockDim.x * threadIdx.y;
	
	if (image_2D_pos.x >= numCols || image_2D_pos.y >= numRows)
		simg[bid] = 1.0f;
	else
		simg[bid] = d_in[index];
	
	__syncthreads();

	for (unsigned int s = blockDim.x*blockDim.y / 2; s > 0; s >>= 1) {
		if (bid < s)
			simg[bid] = min(simg[bid], simg[bid + s]);
		
		__syncthreads();
	}

	if (bid == 0) {
		d_out[blockIdx.x + gridDim.x * blockIdx.y] = simg[0];
	}
}

__global__
void bin_counter(float *d_in,
								 const int numRows, const int numCols,
								 const float minimum, const float range, const int numBins,
								 unsigned int *d_bins)
{
	//extern __shared__ unsigned int s_bin[];
	const int2 image_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
																			 blockIdx.y * blockDim.y + threadIdx.y);
	const int index = image_2D_pos.x + numCols * image_2D_pos.y;

	if (image_2D_pos.x >= numCols || image_2D_pos.y >= numRows)
		return;

	unsigned int bin_index = min(static_cast<unsigned int>(numBins-1),
															 static_cast<unsigned int>((d_in[index] - minimum) / range * numBins));
	atomicAdd(&(d_bins[bin_index]), 1);
}

int main(int argc, char **argv)
{
	// TODO, use dynamic array :'v
	const int ARRAY_SIZE = 5;
	const int ARRAY_BYTES = ARRAY_SIZE * ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
	float h_in[ARRAY_SIZE][ARRAY_SIZE];
	float maxi = -1.0f, mini = 1.0f;
	
	for(int i = 0; i < ARRAY_SIZE; i++) {
		for (int j = 0; j < ARRAY_SIZE; j++) {
			// generate random float in [-1.0f, 1.0f]
			h_in[i][j] = -1.0f + (float)rand()/((float)RAND_MAX/2.0f);
			maxi = max(maxi, h_in[i][j]);
			mini = min(mini, h_in[i][j]);
		}
	}
	printf("Expected max: %f", maxi);
	printf("\n");
	printf("Expected min: %f", mini);
	printf("\n");
	
	for (int i =0; i < ARRAY_SIZE; i++) {
		for (int j = 0; j < ARRAY_SIZE; j++) {
			printf("%f", h_in[i][j]);
			printf("  ");
		}
		printf("\n");
	}

	const int numBins = 5;
	unsigned int h_bins[numBins];
	for (int i = 0; i<numBins; i++) h_bins[i] = 0;

	// guess image is no bigger than 1024*1024
	// declare GPU memory pointers
	float * d_in, * d_inter, * d_max, * d_min;
	unsigned int * d_bins;

	const int size = 4;
	const dim3 blockSize(size, size, 1);
	const dim3 gridSize((int) ARRAY_SIZE/size + 1, (int) ARRAY_SIZE/size + 1, 1);

	int size_first = sizeof(float) * blockSize.x * blockSize.y;
	int size_inter = sizeof(float) * gridSize.x * gridSize.y;
	// allocate GPU memory
	cudaMalloc((void **) &d_in, ARRAY_BYTES);
	cudaMalloc((void **) &d_inter, gridSize.x * gridSize.y * sizeof(float));
	cudaMalloc((void **) &d_min, sizeof(float));
	cudaMalloc((void **) &d_max, sizeof(float));
	cudaMalloc((void **) &d_bins, numBins * sizeof(unsigned int));

	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bins, h_bins, numBins*sizeof(unsigned int), cudaMemcpyHostToDevice);

	reduce_max<<< gridSize, blockSize, size_first >>>
		(d_in,
		 ARRAY_SIZE, ARRAY_SIZE,
		 d_inter);
	cudaDeviceSynchronize();
	reduce_max<<< (1,1,1), gridSize , size_inter >>>
		(d_inter,
		 ARRAY_SIZE/size+1, ARRAY_SIZE/size+1,
		 d_max);
	cudaDeviceSynchronize();

	cudaMemset(d_inter, 0, gridSize.x*gridSize.y*sizeof(float));

	reduce_min<<< gridSize, blockSize, size_first >>>
		(d_in,
		 ARRAY_SIZE, ARRAY_SIZE,
		 d_inter);
	cudaDeviceSynchronize();
	reduce_min<<< (1,1,1), gridSize , size_inter >>>
		(d_inter,
		 ARRAY_SIZE/size+1, ARRAY_SIZE/size+1,
		 d_min);
	cudaDeviceSynchronize();

	// copy back the sum from GPU
	float h_max, h_min;
	cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("Max value: %f", h_max);
	printf("\n");
	printf("Min value: %f", h_min);
	printf("\n");

	bin_counter<<< gridSize, blockSize, numBins*sizeof(unsigned int) >>>
		(d_in,
		 ARRAY_SIZE, ARRAY_SIZE,
		 h_min, h_max-h_min, numBins,
		 d_bins);
	cudaDeviceSynchronize();
	cudaMemcpy(h_bins, d_bins, numBins*sizeof(unsigned int), cudaMemcpyDeviceToHost);

	unsigned int bins_example[numBins];
	for (int i = 0; i<numBins; i++) bins_example[i] = 0;
	
	for (size_t i = 0; i < ARRAY_SIZE; ++i) {
		for (size_t j = 0; j < ARRAY_SIZE; ++j) {
			unsigned int bin = min(static_cast<unsigned int>(numBins - 1),
														 static_cast<unsigned int>((h_in[i][j] - mini) / (maxi-mini) * numBins));
			bins_example[bin]++;
		}
  }

	printf("expected bins \n");
	for (int i =0; i < numBins; i++) {
		printf("%d", bins_example[i]);
		printf("\t");
	}
	printf("\n");

	printf("actual bin \n");
	for (int i =0; i < numBins; i++) {
		printf("%d", h_bins[i]);
		printf("\t");
	}
	printf("\n");
	
	// free GPU memory allocation
	cudaFree(d_in);
	cudaFree(d_inter);
	cudaFree(d_min);
	cudaFree(d_max);
	cudaFree(d_bins);
        
	return 0;
}
