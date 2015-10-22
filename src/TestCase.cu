#include "includes/TestCase.h"

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
#include "includes/Utility.h"

std::vector<float> TestCase::SingleCPUConvolutionOperation(float * h_input_data,
		float * h_kernel, int inputFeaturemapHeight, int inputFeaturemapWidth,
		int kernelHeight, int kernelWidth) {

	std::vector<std::vector<float> > matrix = Utility::ArrayToMatrix(
			h_input_data, inputFeaturemapHeight, inputFeaturemapWidth);

	int kernelIndex = 0;
	float windowValue = 0;

	std::vector<float> cpuValues;

	for (int y = 0; y < inputFeaturemapHeight - kernelHeight + 1; y++) {
		for (int x = 0; x < inputFeaturemapWidth - kernelWidth + 1; x++) {

			windowValue = 0;
			kernelIndex = 0;

			for (int windowY = y; windowY < y + kernelHeight; windowY++) {
				for (int windowX = x; windowX < x + kernelWidth; windowX++) {

					windowValue += matrix.at(windowY).at(windowX)
							* h_kernel[kernelIndex];
					kernelIndex++;

				}
			}
			cpuValues.push_back(windowValue);
		}
	}

	return cpuValues;

}

void TestCase::CPUConvolutionOperation(float * h_input_data, float * h_kernel,
		float * h_bias, float * h_GPU_data, int inputFeatureHeight,
		int inputFeaturemapWidth, int kernelHeight, int kernelWidth,
		int inputFeaturemaps, int outputFeaturemaps, cudnnOutputDim outputDim) {

	int batchSize = 1;

	//参数显示
	std::cout << " output images (batch size) : " << batchSize
			<< " | output featuremaps for each image : "
			<< outputDim.outputFeaturemapsForEachImage
			<< " | output featuremap height : "
			<< outputDim.outputFeaturemapHeight
			<< " | output featuremap width : "
			<< outputDim.outputFeaturemapWidth << std::endl;

	//数组分割
	std::vector<float *> all_inputs = Utility::SplitArray(h_input_data,
			batchSize * inputFeaturemaps,
			inputFeatureHeight * inputFeaturemapWidth);

	std::vector<float *> all_outputs = Utility::SplitArray(h_GPU_data,
			outputDim.outputImages * outputDim.outputFeaturemapsForEachImage,
			outputDim.outputFeaturemapHeight * outputDim.outputFeaturemapWidth);

	std::vector<float *> all_kernels = Utility::SplitArray(h_kernel,
			outputFeaturemaps * inputFeaturemaps, kernelHeight * kernelWidth);

	std::cout << " input elements : " << all_inputs.size()
			<< " | output elements : " << all_outputs.size() << " | kernels : "
			<< all_kernels.size() << std::endl;

	//将输出到下一层同一个节点的卷积运算结果分到同一组
	std::vector<std::vector<std::vector<float>>>groupCpuValues;

	for (int i = 1; i <= outputFeaturemaps; i++) {

		std::vector<std::vector<float> > cpuValues;
		int featuremapIndex = 0;

		for (int j = (i - 1) * inputFeaturemaps; j < i * inputFeaturemaps;
				j++) {

			cpuValues.push_back(
					TestCase::SingleCPUConvolutionOperation(
							all_inputs.at(featuremapIndex), all_kernels.at(j),
							inputFeatureHeight, inputFeaturemapWidth,
							kernelHeight, kernelWidth));
		}

		groupCpuValues.push_back(cpuValues);

	}

	float cpuValue;

	//将输出到下一层同一个节点的卷积运算结果相加
	for (int group = 0; group < groupCpuValues.size(); group++) {

		for (int featuremapIndex = 0;
				featuremapIndex < groupCpuValues.at(group).at(0).size();
				featuremapIndex++) {

			cpuValue = 0;

			//合并同一group的元素
			for (int groupElement = 0;
					groupElement < groupCpuValues.at(group).size();
					groupElement++) {

				cpuValue += groupCpuValues.at(group).at(groupElement).at(
						featuremapIndex);
			}

			//加上偏置项
			cpuValue += h_bias[group];

			std::cout << " conv [ " << featuremapIndex << " ] | CPU value : "
					<< cpuValue << " | GPU value : "
					<< all_outputs.at(group)[featuremapIndex]
					<< " | Is Equal : "
					<< Utility::FloatIsEqual(cpuValue,
							all_outputs.at(group)[featuremapIndex])
					<< std::endl;
		}
	}

}

void TestCase::SingleCPUMaxPoolingOperation(
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

			std::cout << " mxpo [ " << y << " " << x << " ] |"
					<< " CPU value : " << maxValue << " | GPU value : "
					<< gpuData[gpuIndex] << " | Is Equal : "
					<< Utility::FloatIsEqual(maxValue, gpuData[gpuIndex])
					<< std::endl;
			gpuIndex++;

		}

	}

	std::cout << "--- END TestbedOfMaxPoolingMethodForOneOutputFeaturemap ---"
			<< std::endl;

}
