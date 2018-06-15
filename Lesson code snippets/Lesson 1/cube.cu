#include <stdio.h>

__global__
void cube(float * d_out, float * d_in, int size){
	// Todo: Fill in this function
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int index = size * iy + ix;
	float f = d_in[index];
	d_out[index] = f*f*f;
	
}

int main(int argc, char ** argv) {
	// Gonna try a 2d grid size and block size :D, [5,5]
	const int ARRAY_SIZE = 7;
	const int ARRAY_BYTES = ARRAY_SIZE * ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
	float h_in[ARRAY_SIZE][ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++) {
		for (int j = 0; j < ARRAY_SIZE; j++) {
			h_in[i][j] = float(ARRAY_SIZE * i + j);
		}
	}
	float h_out[ARRAY_SIZE][ARRAY_SIZE];

	// declare GPU memory pointers
	float * d_in;
	float * d_out;

	// allocate GPU memory
	cudaMalloc((void**) &d_in, ARRAY_BYTES);
	cudaMalloc((void**) &d_out, ARRAY_BYTES);

	// transfer the array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// launch the kernel
	const int size = 2;
	const dim3 gridSize(ARRAY_SIZE/size + 1, ARRAY_SIZE/size + 1, 1);
	const dim3 blockSize(size, size, 1);
	cube<<<gridSize, blockSize>>>(d_out, d_in, ARRAY_SIZE);

	// copy back the result array to the CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// print out the resulting array
	for (int i =0; i < ARRAY_SIZE; i++) {
		for (int j = 0; j < ARRAY_SIZE; j++) {
			printf("%f", h_out[i][j]);
			printf("\t");
		}
		printf("\n");
	}

	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
	
}