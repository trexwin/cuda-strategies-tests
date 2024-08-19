#define HASMAIN

#include <string>

#include "./Measurements.cu"
#include "./Base_Case.cu"

#include "./Warp_1D_Volatile_Case.cu"
#include "./Warp_2D32_Case.cu"
#include "./Warp_2D64_Case.cu"

Measurements* testFold(const char* name, int iterations, void(*fold)(Measurements*))
{
	Measurements* measurements = measurements_new(name);
	fold(measurements);
	printMeasurements(measurements);
	
	return measurements;
}
 
int main(void) 
{
	// Initialise CUDA. https://stackoverflow.com/questions/15166799/any-particular-function-to-initialize-gpu-other-than-the-first-cudamalloc-call
	cudaFree(0);
	
	// Print the header if csv, otherwise environmental data
	#ifdef CSV
	printCSVHeader();
	#endif
	#ifndef CSV
	printParameters();
	#endif
	
	char baseName[] = "Base_Case";
	free(testFold((char*)&baseName, ITERATIONS, baseCase));
	
	// Warp reductions
	char warp2D32Name[] = "Warp_Volatile_2D32";
	free(testFold((char*)&warp2D32Name, ITERATIONS, warp2D32));
	char warp2D64Name[] = "Warp_Volatile_2D64";
	free(testFold((char*)&warp2D64Name, ITERATIONS, warp2D64));
	char warp1DVolatileName[] = "Warp_Volatile_1D";
	free(testFold((char*)&warp1DVolatileName, ITERATIONS, warp1DVolatile));
	char warp1DSyncThrName[] = "Warp_Syncthreads_1D";
	free(testFold((char*)&warp1DSyncThrName, ITERATIONS, warp1DVolatile));
	char warp1DSyncWarName[] = "Warp_Syncwarp_1D";
	free(testFold((char*)&warp1DSyncWarName, ITERATIONS, warp1DVolatile));
    
	return 0;
}

