#define HASMAIN

#include <string>

#include "./Measurements.cu"
#include "./Base_Case.cu"
#include "./Global_Double_Kernel_Case.cu"
#include "./Global_Multi_Kernel_Case.cu"
#include "./Global_Sync_Case.cu"

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
	
    // Base case, mostly ignored as we are trying pure GPU folds, but data might be usefull
	char baseCaseName[] = "Base";
	free(testFold((char*)&baseCaseName, ITERATIONS, baseCase));
    
	//Prefold cases
	char globalDoubleKernelName[] = "Global_Double_Kernel";
	free(testFold((char*)&globalDoubleKernelName, ITERATIONS, globalDoubleKernel));
	char globalMultiKernelName[] = "Global_Multi_Kernel";
	free(testFold((char*)&globalMultiKernelName, ITERATIONS, globalMultiKernel));
	char globalSyncName[] = "Global_Sync";
	free(testFold((char*)&globalSyncName, ITERATIONS, globalSync));
    
	return 0;
}

