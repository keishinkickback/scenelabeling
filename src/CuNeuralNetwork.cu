#include "CuNeuralNetwork.h"

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

void CuNeuralNetwork::initializeConvolutionalLayerTensorDescriptor(
		cudnnHandle_t *cudnnHandle,
		cudnnTensorDescriptor_t * inputDataTensorDescriptor,
		cudnnFilterDescriptor_t * kernelDescriptor,
		cudnnConvolutionDescriptor_t * convolutionDescriptor,
		cudnnTensorDescriptor_t * outputDataTensorDescriptor,
		cudnnConvolutionFwdAlgo_t * algorithm, int executeBatchSize,
		int imageHeight, int imageWidth, int kernelHeight, int kernelWidth,
		int inputFeaturemaps, int outputFeaturemaps,
		size_t * workspaceSizeInByte, int * outputImages,
		int * outputFeaturemapsForEachImage, int * outputFeaturemapHeight,
		int * outputFeaturemapWidth) {

	//输入数据设定
	checkCUDNN(cudnnCreateTensorDescriptor(inputDataTensorDescriptor));
	checkCUDNN(
			cudnnSetTensor4dDescriptor(*inputDataTensorDescriptor,
					CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, executeBatchSize,
					inputFeaturemaps, imageHeight, imageWidth));

	//卷积核设定
	checkCUDNN(cudnnCreateFilterDescriptor(kernelDescriptor));
	checkCUDNN(
			cudnnSetFilter4dDescriptor(*kernelDescriptor, CUDNN_DATA_FLOAT,
					outputFeaturemaps, inputFeaturemaps, kernelHeight,
					kernelWidth));

	//卷积操作设定
	checkCUDNN(cudnnCreateConvolutionDescriptor(convolutionDescriptor));
	//零填充的行数与列数：0 卷积的水平和垂直的滑动长度：1 x，y向上取样的比例尺：1
	//不使用卷积操作，因为卷积操作要旋转卷积核，而互相关操作无需旋转卷积核
	checkCUDNN(
			cudnnSetConvolution2dDescriptor(*convolutionDescriptor, 0, 0, 1, 1,
					1, 1, CUDNN_CROSS_CORRELATION));

	//输出数据设定
	//获取：图片数量，输出featuremap数量，featuremap的高度，featuremap的宽度

	checkCUDNN(
			cudnnGetConvolution2dForwardOutputDim(*convolutionDescriptor,
					*inputDataTensorDescriptor, *kernelDescriptor, outputImages,
					outputFeaturemapsForEachImage, outputFeaturemapHeight,
					outputFeaturemapWidth));

	checkCUDNN(cudnnCreateTensorDescriptor(outputDataTensorDescriptor));
	checkCUDNN(
			cudnnSetTensor4dDescriptor(*outputDataTensorDescriptor,
					CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, *outputImages,
					*outputFeaturemapsForEachImage, *outputFeaturemapHeight,
					*outputFeaturemapWidth));

	//选择FP算法
	checkCUDNN(
			cudnnGetConvolutionForwardAlgorithm(*cudnnHandle,
					*inputDataTensorDescriptor, *kernelDescriptor,
					*convolutionDescriptor, *outputDataTensorDescriptor,
					CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algorithm));

	//获取workspace的大小
	checkCUDNN(
			cudnnGetConvolutionForwardWorkspaceSize(*cudnnHandle,
					*inputDataTensorDescriptor, *kernelDescriptor,
					*convolutionDescriptor, *outputDataTensorDescriptor,
					*algorithm, workspaceSizeInByte));

}

void CuNeuralNetwork::syncTrainingDataToDevice(float * h_input_data,
		float * d_input_data, float * h_kernel, float * d_kernel,
		float * d_output_data, void * d_cudnn_workspace, int executeBatchSize,
		int imageHeight, int imageWidth, int kernelHeight, int kernelWidth,
		int inputFeaturemaps, int outputFeaturemaps, int outputImages,
		int outputFeaturemapsForEachImage, int outputFeaturemapHeight,
		int outputFeaturemapWidth, size_t workspaceSizeInByte) {

	//输入层数据同步
	checkCudaErrors(
			cudaMalloc(&d_input_data,
					sizeof(float) * executeBatchSize * inputFeaturemaps
							* imageHeight * imageWidth));
	checkCudaErrors(
			cudaMemcpyAsync(d_input_data, h_input_data,
					sizeof(float) * executeBatchSize * inputFeaturemaps
							* imageHeight * imageWidth,
					cudaMemcpyHostToDevice));

	//卷积核数据同步
	checkCudaErrors(
			cudaMalloc(&d_kernel,
					sizeof(float) * outputFeaturemaps * kernelHeight
							* kernelWidth));
	checkCudaErrors(
			cudaMemcpyAsync(d_kernel, h_kernel,
					sizeof(float) * outputFeaturemaps * kernelHeight
							* kernelWidth, cudaMemcpyHostToDevice));

	//输出层数据同步
	//输出层只做空间分配，不做数据同步

	checkCudaErrors(
			cudaMalloc(&d_output_data,
					sizeof(float) * outputImages * outputFeaturemapsForEachImage
							* outputFeaturemapHeight * outputFeaturemapWidth));

	std::cout << outputImages << std::endl;

	//workspace空间分配
	checkCudaErrors(cudaMalloc(&d_cudnn_workspace, workspaceSizeInByte));

	checkCudaErrors(cudaDeviceSynchronize());

}
