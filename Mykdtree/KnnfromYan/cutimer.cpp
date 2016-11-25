#include "cutimer.h"
#include "cuda_runtime.h"
#include "stdio.h"

void CuTimer::startTimer()
{
    cudaEventCreate( &start);
	cudaEventCreate( &stop );

	cudaEventRecord( start, 0 ) ;
}

void CuTimer::finishTimer(char* content)
{
	cudaEventRecord( stop, 0 ) ;
	cudaEventSynchronize( stop) ;
	cudaEventElapsedTime( &elapsedTime,	start, stop );
	printf("%s Timer using: %f ms\n", content, elapsedTime);

	cudaEventDestroy( start );
    cudaEventDestroy( stop );
}