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
#include <math.h>

//////////////////////////////////////////////////////////////////////////////
// Error handling
// Adapted from the CUDNN classification code
// sample: https://developer.nvidia.com/cuDNN

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

//将一维vector转换为普通一维动态数组
float* Utility::VectorToDynamicArray(std::vector<float> * input) {

	float * array;
	array = new float[input->size()];

	for (int i = 0; i < input->size(); i++) {
		array[i] = input->at(i);
	}

	return array;
}

//打印动态数组
void Utility::PrintDynamicArray(float * array, int length) {
	for (int i = 0; i < length; i++) {
		std::cout << array[i] << std::endl;
	}
}

//将一维数组根据设定的行数与列数转换为矩阵
//矩阵使用二维向量描述
std::vector<std::vector<float> > Utility::ArrayToMatrix(float * array, int row,
		int column) {

	int arrayIndex = 0;
	std::vector<std::vector<float> > matrix;

	for (int y = 0; y < row; y++) {

		std::vector<float> row;

		for (int x = 0; x < column; x++) {

			row.push_back(array[arrayIndex]);
			arrayIndex++;

		}

		matrix.push_back(row);

	}

	return matrix;

}

//在单精度有损运算中，判断2个float值是否相等，当两值的差小于等于0.00001的时候判断两者相等
bool Utility::FloatIsEqual(float a, float b) {

	if (abs(a - b) <= 0.00001f) {

		return true;

	} else {

		return false;

	}

}

//根据设定的分割份数和步长切割数组
std::vector<float *> Utility::SplitArray(float * array, int split,
		int stepLength) {

	std::vector<float *> column;

	//切割成split分
	for (int i = 1; i <= split; i++) {

		std::vector<float> row;

		for (int j = (i - 1) * stepLength; j < i * stepLength; j++) {
			row.push_back(array[j]);
		}

		column.push_back(Utility::VectorToDynamicArray(&row));
	}

	return column;

}

//z-score归一化
void Utility::ZScoreNormalization(std::vector<float> &vector) {

	int length = vector.size();

	//求均值
	float sum = 0;
	for (int i = 1; i < length; i++) {
		sum += vector[i];
	}
	float mean = sum / length;

	//求标准差
	float standardDeviation = 0;
	for (int i = 1; i < length; i++) {
		standardDeviation += vector[i] - mean;
	}
	standardDeviation = sqrt(standardDeviation / length);

	//归一化
	for (int i = 1; i < length; i++) {
		vector[i] = (vector[i] - mean) / standardDeviation;
	}

}

//min-max归一化
void Utility::MinMaxNormalization(std::vector<float> &vector) {

	//最小值
	float minValue = *std::min_element(vector.begin(), vector.end());

	//最大值
	float maxValue = *std::max_element(vector.begin(), vector.end());

	//归一化
	for (int i = 0; i < vector.size(); i++) {
		vector[i] = (vector[i] - minValue) / (maxValue - minValue);
	}

}

//为host数据创建统一寻址的内存，如果host数据不为空，需要同步数据到统一内存地址。
float * Utility::AllocUnifiedMemory(float * data, int dataLength,
		bool needSync = false) {

	//内存地址长度
	int memoryLength = dataLength * sizeof(float);

	//统一寻址地址
	float * u_data;
	checkCudaErrors(cudaMallocManaged(&u_data, memoryLength));

	//如果需要同步数据
	if (NULL != data && needSync == true) {
		memcpy(u_data, data, memoryLength);

	}

	return u_data;
}

//将host数据拷贝到device
float * Utility::HostToDevice(float * h_data, int length) {

	float * d_data;
	checkCudaErrors(cudaMalloc(&d_data, sizeof(float) * length));
	checkCudaErrors(
			cudaMemcpyAsync(d_data, h_data, sizeof(float) * length,
					cudaMemcpyHostToDevice));
	return d_data;

}

//将device的数据拷贝到host
float * DeviceToHost(float * d_data, int length) {

	float * h_data;
	checkCudaErrors(
			cudaMemcpyAsync(h_data, d_data, sizeof(float) * length,
					cudaMemcpyDeviceToHost));
	return h_data;

}
