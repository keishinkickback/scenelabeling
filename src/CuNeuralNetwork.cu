#include "includes/CuNeuralNetwork.h"

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

float * CuNeuralNetwork::initializeInputDataLayer(float * h_input_data,
		cudnnTensorDescriptor_t * inputDataTensorDescriptor, int batchSize,
		int inputFeaturemaps, int imageHeight, int imageWidth) {

	checkCUDNN(cudnnCreateTensorDescriptor(inputDataTensorDescriptor));
	checkCUDNN(
			cudnnSetTensor4dDescriptor(*inputDataTensorDescriptor,
					CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize,
					inputFeaturemaps, imageHeight, imageWidth));

	float * d_input_data;

	checkCudaErrors(
			cudaMalloc(&d_input_data,
					sizeof(float) * batchSize * inputFeaturemaps * imageHeight
							* imageWidth));

	checkCudaErrors(
			cudaMemcpyAsync(d_input_data, h_input_data,
					sizeof(float) * batchSize * inputFeaturemaps * imageHeight
							* imageWidth, cudaMemcpyHostToDevice));

	return d_input_data;

}

float * CuNeuralNetwork::initializeKernels(float * h_kernel,
		cudnnFilterDescriptor_t * kernelDescriptor, int inputFeaturemaps,
		int outputFeaturemaps, int kernelHeight, int kernelWidth) {

	checkCUDNN(cudnnCreateFilterDescriptor(kernelDescriptor));
	checkCUDNN(
			cudnnSetFilter4dDescriptor(*kernelDescriptor, CUDNN_DATA_FLOAT,
					outputFeaturemaps, inputFeaturemaps, kernelHeight,
					kernelWidth));
	float *d_kernel;
	checkCudaErrors(
			cudaMalloc(&d_kernel,
					sizeof(float) * outputFeaturemaps * inputFeaturemaps
							* kernelHeight * kernelWidth));
	checkCudaErrors(
			cudaMemcpyAsync(d_kernel, h_kernel,
					sizeof(float) * outputFeaturemaps * inputFeaturemaps
							* kernelHeight * kernelWidth,
					cudaMemcpyHostToDevice));
	return d_kernel;

}

float * CuNeuralNetwork::initializeOutputDataLayer(
		cudnnTensorDescriptor_t * inputDataTensorDescriptor,
		cudnnFilterDescriptor_t * kernelDescriptor,
		cudnnConvolutionDescriptor_t * convolutionDescriptor,
		cudnnTensorDescriptor_t * outputDataTensorDescriptor,
		cudnnOutputDim * outputDim) {

	checkCUDNN(
			cudnnGetConvolution2dForwardOutputDim(*convolutionDescriptor,
					*inputDataTensorDescriptor, *kernelDescriptor,
					&outputDim->outputImages,
					&outputDim->outputFeaturemapsForEachImage,
					&outputDim->outputFeaturemapHeight,
					&outputDim->outputFeaturemapWidth));

	checkCUDNN(cudnnCreateTensorDescriptor(outputDataTensorDescriptor));
	checkCUDNN(
			cudnnSetTensor4dDescriptor(*outputDataTensorDescriptor,
					CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
					outputDim->outputImages,
					outputDim->outputFeaturemapsForEachImage,
					outputDim->outputFeaturemapHeight,
					outputDim->outputFeaturemapWidth));
	//device上分配内存空间
	float *d_output_data;
	checkCudaErrors(
			cudaMalloc(&d_output_data,
					sizeof(float) * outputDim->outputImages
							* outputDim->outputFeaturemapsForEachImage
							* outputDim->outputFeaturemapHeight
							* outputDim->outputFeaturemapWidth));

	return d_output_data;

}

float * CuNeuralNetwork::addBiasUnits(float * h_bias,
		cudnnTensorDescriptor_t * biasTensorDescriptor, int outputFeaturemaps,
		int kernelHeight, int kernelWidth) {

	//偏置项设定
	checkCUDNN(cudnnCreateTensorDescriptor(biasTensorDescriptor));
	checkCUDNN(
			cudnnSetTensor4dDescriptor(*biasTensorDescriptor, CUDNN_TENSOR_NCHW,
					CUDNN_DATA_FLOAT, 1, outputFeaturemaps, 1, 1));

	//分配内存
	float *d_bias;
	checkCudaErrors(
			cudaMalloc(&d_bias,
					sizeof(float) * outputFeaturemaps * kernelHeight
							* kernelWidth));
	checkCudaErrors(
			cudaMemcpyAsync(d_bias, h_bias,
					sizeof(float) * outputFeaturemaps * kernelHeight
							* kernelWidth, cudaMemcpyHostToDevice));

	return d_bias;

}

float * CuNeuralNetwork::initializePoolingLayer(float * d_output_data,
		cudnnTensorDescriptor_t * inputDataTensorDescriptor,
		cudnnPoolingDescriptor_t * poolingDescriptor,
		cudnnTensorDescriptor_t * poolingDataTensorDescriptor,
		cudnnOutputDim * outputDim, int poolingWindowHeight,
		int poolingWindowWidth, int poolingVerticalStride,
		int poolingHorizontalStride, cudnnOutputDim * poolingOutputDim) {

	float * d_pooling_output_data;
	int poolingOutputDataHeight = (outputDim->outputFeaturemapHeight
			- poolingWindowHeight) / poolingHorizontalStride + 1;
	int poolingOutputDataWidth = (outputDim->outputFeaturemapWidth
			- poolingWindowWidth) / poolingVerticalStride + 1;

	checkCudaErrors(
			cudaMalloc(&d_pooling_output_data,
					sizeof(float) * outputDim->outputImages
							* outputDim->outputFeaturemapsForEachImage
							* poolingOutputDataHeight
							* poolingOutputDataWidth));

	checkCUDNN(cudnnCreatePoolingDescriptor(poolingDescriptor));
	checkCUDNN(
			cudnnSetPooling2dDescriptor(*poolingDescriptor, CUDNN_POOLING_MAX,
					poolingWindowHeight, poolingWindowWidth, 0, 0,
					poolingVerticalStride, poolingHorizontalStride));

	//没有实装这个方法？
//	checkCUDNN(
//			cudnnGetPooling2dForwardOutputDim(
//					*poolingDescriptor,
//					*inputDataTensorDescriptor,
//					&poolingOutputDim->outputImages,
//					&poolingOutputDim->outputFeaturemapsForEachImage,
//					&poolingOutputDim->outputFeaturemapHeight,
//					&poolingOutputDim->outputFeaturemapWidth));
//
//	std::cout << poolingOutputDim->outputImages << " "
//			<< poolingOutputDim->outputFeaturemapsForEachImage << " "
//			<< poolingOutputDim->outputFeaturemapHeight << " "
//			<< poolingOutputDim->outputFeaturemapWidth << std::endl;

	poolingOutputDim->outputImages = outputDim->outputImages;
	poolingOutputDim->outputFeaturemapsForEachImage =
			outputDim->outputFeaturemapsForEachImage;
	poolingOutputDim->outputFeaturemapHeight = poolingOutputDataHeight;
	poolingOutputDim->outputFeaturemapWidth = poolingOutputDataWidth;

	checkCUDNN(cudnnCreateTensorDescriptor(poolingDataTensorDescriptor));
	checkCUDNN(
			cudnnSetTensor4dDescriptor(*poolingDataTensorDescriptor,
					CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
					outputDim->outputImages,
					outputDim->outputFeaturemapsForEachImage,
					poolingOutputDataHeight, poolingOutputDataWidth));

	return d_pooling_output_data;

}

float * computeFullconnectedDataLayer(
		cublasHandle_t cublasHandle,
		float * d_input_data,
		float * d_kernel,
		float * d_bias,
		float * d_ones_vector,
		int outputFeaturemapLength,
		int inputFeaturemaps,
		int inputFeaturemapHeight, int inputFeaturemapWidth) {

	float * d_output_data;

//	int featuremapsLength = inputFeaturemapHeight * inputFeaturemapWidth
//			* inputFeaturemaps;
//
//	checkCudaErrors(
//			cudaMalloc((void** ) &d_output_data,
//					outputFeaturemaps * sizeof(float)));
//
//	checkCudaErrors(
//			cudaMemset(d_output_data, 0,
//					outputFeaturemapLength * sizeof(float)));
//
//	float alpha = 1.0;
//	float beta = 0;
//
//	//全连接运算
//	checkCudaErrors(
//			cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, outputFeaturemaps, 1, featuremapsLength, &alpha, d_input_data, featuremapsLength, d_kernel, featuremapsLength, &beta, d_output_data, outputFeaturemaps));
//
//	//加上偏置项
//	//d_ones_vector应为1*1
//	beta = 1;
//	checkCudaErrors(
//			cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, outputFeaturemaps, 1, 1, &alpha, d_bias, featuremapsLength, d_ones_vector, featuremapsLength, &beta, d_output_data, outputFeaturemaps));

	return d_output_data;

}
