#include "./Measurements.cu"
#include "./Pragmas.cu"

// This uses the volatile technique, but I'm too lazy to update all names
__device__ void warp2D32XWarp(volatile int* sdata, const unsigned int tx){
	sdata[tx] += sdata[tx+16];
	sdata[tx] += sdata[tx+8];
	sdata[tx] += sdata[tx+4];
	sdata[tx] += sdata[tx+2];
	sdata[tx] += sdata[tx+1];
}

__global__ void warp2D32Kernel(const int *input, int* output)
{
	__shared__ int sdata[THREADBLOCKSIZE/32][32];
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;
	const unsigned int bx = blockIdx.x;
	const size_t offset = bx * THREADBLOCKSIZE + (ty * 32 + tx);
	
	// Read data
	sdata[ty][tx] = offset < INPUTSIZE ? input[offset] : 0;
	__syncthreads();
	
	// Folding
	if(tx < 16) {
		warp2D32XWarp(sdata[ty], tx);
		__syncthreads();
		
		for(int i = THREADBLOCKSIZE/64; i > 0; i >>= 1) {
			BASIC_REDUCTION_Y(sdata, tx, ty, i)
		}
				
		if(tx == 0 && ty == 0)
			output[bx] = sdata[0][0];
	}
}

void warp2D32(Measurements* measurements)
{
    const size_t inputByteSize = INPUTSIZE * sizeof(int);
	const size_t outputByteSize = GRIDSIZE * sizeof(int);
	int *h_input, *d_input, *h_output, *d_output;
	int res = 0, trueRes = 0;
	
	// Initialise input arrays with 1s
	h_input = (int*)mallocMeasurements(inputByteSize, measurements);
	for(size_t i = 0; i < INPUTSIZE; ++i) {
		h_input[i] = i; 
		trueRes += i; 
	}
	
	cudaMalloc(&d_input, INPUTSIZE * sizeof(int));
	cudaMemcpy(d_input, h_input, inputByteSize, cudaMemcpyHostToDevice);
	
	// Initialise output arrays
	h_output = (int*)mallocMeasurements(outputByteSize, measurements);
	cudaMalloc(&d_output, outputByteSize);
	
	// Fold operation
	for(int i = 0; i < ITERATIONS; ++i) {
		startGpuTimer(measurements);
		warp2D32Kernel<<<GRIDSIZE, dim3(32, THREADBLOCKSIZE/32)>>>(d_input, d_output);
		recordTime(measurements, measurements->gpuFoldTime, stopGpuTimer(measurements));
		cudaDeviceSynchronize();
		measurements->iterations++;
	}
	cudaMemcpy(h_output, d_output, outputByteSize, cudaMemcpyDeviceToHost);
	
	// Linear fold on CPU
	for(size_t i = 0; i < GRIDSIZE; ++i)
		res += h_output[i];
	
	// Update final measurements
	measurements->success = measurements->success && res == trueRes;
	checkGpuErrors(measurements);
	
	// Free everything
	free(h_input);
	free(h_output);
	cudaFree(d_input);
	cudaFree(d_output);
}

#ifndef HASMAIN
#define HASMAIN
int main(void)
{
	char foldName[] = "Warp_Volatile_2D32";
	Measurements* measurements = measurements_new((char*)&foldName);
	warp2D32(measurements);
	printMeasurements(measurements);
	
	free(measurements);
	
	return 0;
}
#endif
