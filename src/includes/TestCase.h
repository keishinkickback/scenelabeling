/*
 * TestCase.h
 *
 *  Created on: Aug 21, 2015
 *      Author: root
 */

#ifndef TESTCASE_H_
#define TESTCASE_H_

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
#include "CuNeuralNetwork.h"

class TestCase {

public:

	static std::vector<float> SingleCPUConvolutionOperation(float * inputData,
			float * kernel, int inputFeaturemapHeight, int inputFeaturemapWidth,
			int kernelHeight, int kernelWidth);

	static void SingleCPUMaxPoolingOperation(
			float * inputData, float * gpuData, int inputFeaturemapHeight,
			int inputFeaturemapWidth, int windowHeight, int windowWidth);

	static void CPUConvolutionOperation(float * h_input_data, float * h_kernel,
			float * bias, float * h_GPU_data, int inputFeatureHeight,
			int inputFeaturemapWidth, int kernelHeight, int kernelWidth,
			int inputFeaturemaps, int outputFeaturemaps,
			cudnnOutputDim outputDim);

};

#endif /* TESTCASE_H_ */
