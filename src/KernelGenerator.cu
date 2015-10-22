#include "includes/KernelGenerator.h"
#include "includes/Utility.h"

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

void KernelGenerator::InitializeKernelUsingXavierAlgorithm(int kernelHeight,
		int kernelWeight, int inputFeaturemaps, std::vector<float> * kernel) {

	//随机数生成器初始化
	std::random_device rd;
	//使用马特赛特旋转演算法伪随机数生成器
	std::mt19937 generator(rd());

	//Xavier算法,分母为kernel的输出维度
	//参考：http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1XavierFiller.html
	float scale = sqrt(3.0f / (kernelHeight * kernelWeight * inputFeaturemaps));

	std::uniform_real_distribution<> distribution(-scale, scale);

	for (int i = 0; i < kernelHeight * kernelWeight; i++) {
		kernel->at(i) = static_cast<float>(distribution(generator));
	}

}

void KernelGenerator::InitializeBiasUsingXavierAlgorithm(int inputFeaturemaps,
		std::vector<float> * bias) {

	std::random_device rd;
	std::mt19937 generator(rd());

	float scale = sqrt(3.0f / (inputFeaturemaps));

	std::uniform_real_distribution<> distribution(-scale, scale);

	for (int i = 0; i < bias->size(); i++) {
		bias->at(i) = static_cast<float>(distribution(generator));
	}

}

std::vector<float> KernelGenerator::InitializeKernels(int inputFeaturemaps,
		int outputFeaturemaps, int kernelHeight, int kernelWidth) {

	//卷积核初始化
	std::vector<float> kernels;

	for (int i = 0; i < inputFeaturemaps * outputFeaturemaps; i++) {
		std::vector<float> kernel(kernelHeight * kernelWidth);
		KernelGenerator::InitializeKernelUsingXavierAlgorithm(kernelHeight,
				kernelWidth, inputFeaturemaps, &kernel);
		kernels.insert(kernels.end(), kernel.begin(), kernel.end());
	}

	return kernels;
}
