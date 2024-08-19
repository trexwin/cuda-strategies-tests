# cuda-strategies-tests
Several CUDA program files used to compare different GPU fold strategies during my master thesis. A strategy in this context refers to an interchangeable segment of a folding algorithm. At the top level is a bash file to run the tests, every folder contains the files with the CUDA code for the test.

## bash_run_expirements
This bash script compiles and runs program files of a specific set of strategies. To properly evaluate the performance at different input sizes while not running too many tests, the tests are divided into three size categories.  The exact values associated with these values can be found in `bash_run_expirements`.
- **kB** range
- **MB** range
- **GB** range

To run tests, simply navigate to one of the folders containing the strategies and run the following command in Bash.
```
../bash_run_expirements
```
This generates a folder named `Test_Results` containing several csv files for the different input sizes and one with the results for all input input sizes.

## Global
These are the global strategies and refer to how a GPU fold algorithm folds its partial results into a singular final result. Three strategies are considered that fully fold on the GPU.
1. **Multi kernel**: Repeatedly launches partial fold kernels till only a single partial result is produced.
2. **Single kernel**: Uses the `atomicAdd` function to obtain the last active thread block, which then folds all partial results into a single result.
3. **Double kernel**: Similar to **single kernel** but uses kernel termination as synchronisation. Launches a single thread block kernel afterwards to perform the full fold.

## Prefold
The prefold strategies refer to the method and how many values a single thread folds before folding the entire thread block. 
1. **Loop prefold**: A while loop iterates through the input values and repeatedly folds them with a specific shared memory array index. 
2. **Inline prefold**: An unrolled version of **loop prefold**.

## Fold
Every thread loads in one or more values into a single shared memory array index, the fold strategy defines how a thread block folds its shared memory array into a single value.
1. **For fold**: Folds the shared memory array using a for loop, which repeatedly disables half of the active threads till only a single thread remains.
2. **Unroll fold**: Unrolls the iterations of the **for fold** strategy.
3. **Nested fold**: Nests the unrolled iterations of the **unroll fold** strategy.
4. **2D variants**: A 2D variant for the **unroll fold** and the **nested fold** strategies are also tested.

## Warp
Once the fold strategy enters the 32 thread range, it is within the warp size. Threads within the warp size act in lock-step, allowing for special optimisations. These are referred to as the warp strategies. Only warp strategies that have some form of explicit synchronisation are considered. 
1. **Syncthreads warp**: No longer halves the amount of active threads every iteration, only having a `__synchtreads` function call between folds.
2. **Syncwarp warp**: Replaces the `__synchtreads` of the **syncthreads warp** strategy with `__synchwarp`.
3. **Volatile warp**: Lifts the warp folding into a separate function where the shared memory array is marked as volatile. This function does not halve the amount of active threads or call `__synchtreads`.
4. **Volatile 2D warp**: 2D variants of the **volatile warp** strategy, one with a 32 by 32 thread block and one of a 64 by 16 thread block.






