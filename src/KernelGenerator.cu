#include "KernelGenerator.h"

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

void KernelGenerator::initializeKernelUsingXavierAlgorithm(int kernelHeight,
		int kernelWeight, int channelNumber, std::vector<float> * kernel) {

	//随机数生成器初始化
	std::random_device rd;
	//使用马特赛特旋转演算法伪随机数生成器
	std::mt19937 generator(rd());

	float core = sqrt(3.0f / (kernelHeight * kernelWeight * channelNumber));

	std::uniform_real_distribution<> distribution(-core, core);

	for (int i = 0; i < kernel->size(); i++) {
		kernel->at(i) = static_cast<float>(distribution(generator));
	}

}

void KernelGenerator::initializeBiasUsingXavierAlgorithm(int kernelHeight,
		int kernelWeight, int channelNumber, std::vector<float> * bias) {

	//随机数生成器初始化
	std::random_device rd;
	//使用马特赛特旋转演算法伪随机数生成器
	std::mt19937 generator(rd());

	float core = sqrt(3.0f / (kernelHeight * kernelWeight * channelNumber));

	std::uniform_real_distribution<> distribution(-core, core);

	for (int i = 0; i < bias->size(); i++) {
		bias->at(i) = static_cast<float>(distribution(generator));
	}

}
