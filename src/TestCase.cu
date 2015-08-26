#include "TestCase.h"

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
#include "Utility.h"

void TestCase::TestbedOfConvolutionMethodForOneOutputFeaturemap(
		float * inputData, float * gpuData, float * kernel, float bias,
		int inputFeaturemapHeight, int inputFeaturemapWidth, int kernelHeight,
		int kernelWidth) {

	std::cout
			<< "--- START TestbedOfConvolutionMethodForOneOutputFeaturemap ---"
			<< std::endl;

	std::vector<std::vector<float> > matrix = Utility::ArrayToMatrix(inputData,
			inputFeaturemapHeight, inputFeaturemapWidth);

	int gpuIndex = 0;
	int kernelIndex, biasIndex;
	float windowValue = 0;

	for (int y = 0; y < inputFeaturemapHeight - kernelHeight + 1; y++) {
		for (int x = 0; x < inputFeaturemapWidth - kernelWidth + 1; x++) {

			kernelIndex = 0;
			biasIndex = 0;
			windowValue = 0;

			for (int windowY = y; windowY < y + kernelHeight; windowY++) {
				for (int windowX = x; windowX < x + kernelWidth; windowX++) {
					windowValue += matrix.at(windowY).at(windowX)
							* kernel[kernelIndex];
					kernelIndex++;
					biasIndex++;
				}
			}

			std::cout << " conv [ " << y << " " << x << " ] |" << " CPU value : "
					<< windowValue + bias << " | GPU value : "
					<< gpuData[gpuIndex] << " | Is Equal : "
					<< Utility::floatIsEqual(windowValue + bias,
							gpuData[gpuIndex]) << std::endl;
			gpuIndex++;

		}

	}

	std::cout << "--- END TestbedOfConvolutionMethodForOneOutputFeaturemap ---"
			<< std::endl;

}

void TestCase::TestbedOfMaxPoolingMethodForOneOutputFeaturemap(
		float * inputData, float * gpuData, int inputFeaturemapHeight,
		int inputFeaturemapWidth, int windowHeight, int windowWidth) {

	std::cout << "--- START TestbedOfMaxPoolingMethodForOneOutputFeaturemap ---"
			<< std::endl;

	std::vector<std::vector<float> > matrix = Utility::ArrayToMatrix(inputData,
			inputFeaturemapHeight, inputFeaturemapWidth);

	int gpuIndex = 0;
	float maxValue = 0;

	for (int y = 0; y < inputFeaturemapHeight - windowHeight + 1; y++) {
		for (int x = 0; x < inputFeaturemapWidth - windowWidth + 1; x++) {

			maxValue = matrix.at(y).at(x);

			for (int windowY = y; windowY < y + windowHeight; windowY++) {
				for (int windowX = x; windowX < x + windowWidth; windowX++) {
					if (matrix.at(windowY).at(windowX) > maxValue) {
						maxValue = matrix.at(windowY).at(windowX);
					}
				}
			}

			std::cout << " mxpo [ " << y << " " << x << " ] |" << " CPU value : "
					<< maxValue << " | GPU value : " << gpuData[gpuIndex]
					<< " | Is Equal : "
					<< Utility::floatIsEqual(maxValue, gpuData[gpuIndex])
					<< std::endl;
			gpuIndex++;

		}

	}

	std::cout << "--- END TestbedOfMaxPoolingMethodForOneOutputFeaturemap ---"
			<< std::endl;

}
