/*
 * Utility.h
 *
 *  Created on: Aug 21, 2015
 *      Author: root
 */

#ifndef UTILITY_H_
#define UTILITY_H_

#include <iostream>
#include <sstream>
#include <numeric>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <FreeImage.h>
#include <unistd.h>
#include <string.h>
#include <algorithm>

class Utility {

public:
	static float * VectorToDynamicArray(std::vector<float> * input);
	static void PrintDynamicArray(float * array, int length);
	static std::vector<std::vector<float> > ArrayToMatrix(float * array,
			int width, int height);
	static bool FloatIsEqual(float a, float b);
	static std::vector<float *> SplitArray(float * array, int part, int step);
	static void ZScoreNormalization(std::vector<float> &vector);
	static void MinMaxNormalization(std::vector<float> &vector);
	static float * AllocUnifiedMemory(float * h_data, int dataLength,
			bool needSync);
	static float * HostToDevice(float * h_data, int length);
	static float * DeviceToHost(float * d_data, int length);
};

#endif /* UTILITY_H_ */
