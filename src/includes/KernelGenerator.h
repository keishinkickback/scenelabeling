/*
 * KernelGenerator.h
 *
 *  Created on: Aug 21, 2015
 *      Author: root
 */

#ifndef KERNELGENERATOR_H_
#define KERNELGENERATOR_H_

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

class KernelGenerator {

public:

	static void InitializeKernelUsingXavierAlgorithm(int kernelHeight,
			int kernelWeight, int inputFeaturemaps,
			std::vector<float> * kernel);

	static void InitializeBiasUsingXavierAlgorithm(int outputFeaturemaps,
			std::vector<float> * bias);

	static std::vector<float> InitializeKernels(int inputFeaturemaps, int outputFeaturemaps,
			int kernelHeight, int kernelWidth);

};

#endif /* KERNELGENERATOR_H_ */
