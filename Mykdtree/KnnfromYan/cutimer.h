#ifndef _CUTIMER_H_
#define _CUTIMER_H_

#include "cuda.h"
#include "driver_types.h"

class CuTimer
{

    cudaEvent_t     start, stop;
	float           elapsedTime;
public:
	void startTimer();
	void finishTimer(char* content = NULL);
};

#endif