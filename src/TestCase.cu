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
		int inputFeatureHeight, int inputFeaturemapWidth, int kernelHeight,
		int kernelWidth) {

	std::vector<std::vector<float> > matrix = Utility::ArrayToMatrix(inputData,
			inputFeatureHeight, inputFeaturemapWidth);

	int gpuIndex = 0;
	int kernelIndex, biasIndex;
	float windowValue = 0;

	for (int y = 0; y < inputFeatureHeight - kernelHeight + 1; y++) {
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

			std::cout << " [ " << y << " " << x << " ] " << " CPU value : "
					<< windowValue + bias << " | GPU value : "
					<< gpuData[gpuIndex] << std::endl;
			gpuIndex++;

		}

	}

}
