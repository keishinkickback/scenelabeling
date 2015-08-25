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

	void initializeKernelUsingXavierAlgorithm(int kernelHeight,
			int kernelWeight, int outputFeaturemaps,
			std::vector<float> * kernel);

	void initializeBiasUsingXavierAlgorithm(int outputFeaturemaps,
			std::vector<float> * bias);

};

#endif /* KERNELGENERATOR_H_ */
