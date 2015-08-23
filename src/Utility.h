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
	static float * VectorToArray(std::vector<float> * input);
	static void PrintDynamicArray(float * array, int length);
	static std::vector<std::vector<float> > ArrayToMartix(float * array,
			int width, int height);
};

#endif /* UTILITY_H_ */
