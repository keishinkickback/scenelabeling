/*
 * CuConvolutionLayer.h
 *
 *  Created on: Oct 22, 2015
 *      Author: ryukinkou@gmail.com
 */

#ifndef CUCONVOLUTIONLAYER_H_
#define CUCONVOLUTIONLAYER_H_

#include <cuda_runtime.h>
#include <cudnn.h>

#include "Utility.h"

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

class CuConvolutionLayer {

private:

	cudnnHandle_t cudnnHandle;

	//输入特征图数量
	int inputFeaturemaps;

	//输入特征图组的数量
	int inputBatchSize;

	//特征图的高度
	int inputFeaturemapHeight;

	//特征图宽度
	int inputFeaturemapWidth;

	//输入数据的配置
	cudnnTensorDescriptor_t inputDataTensorDescriptor;

	int kernelHeight;
	int kernelWidth;
	cudnnFilterDescriptor_t kernelTensorDescriptor;

	int zeroPaddingHeight;
	int zeroPaddingWidth;
	int verticalFilterStride;
	int horizontalFilterStride;
	int upscaleInXDirection;
	int upscaleInYDirection;
	cudnnConvolutionDescriptor_t convolutionDescriptor;

	int outputFeaturemaps;
	int outputBatchSize;
	int outputFeaturemapHeight;
	int outputFeaturemapWidth;
	cudnnTensorDescriptor_t outputDataTensorDescriptor;

	//卷积函数
	cudnnConvolutionFwdAlgo_t convolutionForwardAlgorithm;

	//工作区域指针
	void *d_cudnn_workspace;

	//预订工作区域大小，0为无限制
	size_t workspaceSizeInByte;

	//初始化输入数据配置
	void initializeInputDataTensorDescriptor() {
		checkCUDNN(
				cudnnCreateTensorDescriptor(&this->inputDataTensorDescriptor));
		checkCUDNN(
				cudnnSetTensor4dDescriptor(this->inputDataTensorDescriptor,
						CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
						this->inputBatchSize, this->inputFeaturemaps,
						this->inputFeaturemapHeight,
						this->inputFeaturemapWidth));
	}

	//初始化卷积核配置
	void initializeKernelTensorDescriptor() {
		checkCUDNN(cudnnCreateFilterDescriptor(&this->kernelTensorDescriptor));
		checkCUDNN(
				cudnnSetFilter4dDescriptor(this->kernelTensorDescriptor,
						CUDNN_DATA_FLOAT, this->outputFeaturemaps,
						this->inputFeaturemaps, this->kernelHeight,
						this->kernelWidth));
	}

	//初始化卷积操作配置
	void initializeConvolutionDescriptor() {

		checkCUDNN(
				cudnnCreateConvolutionDescriptor(&this->convolutionDescriptor));
		//不使用卷积操作，因为卷积操作要旋转卷积核，而这里无需旋转。
		checkCUDNN(
				cudnnSetConvolution2dDescriptor(this->convolutionDescriptor,
						this->zeroPaddingHeight, this->zeroPaddingWidth,
						this->verticalFilterStride,
						this->horizontalFilterStride, this->upscaleInXDirection,
						this->upscaleInYDirection, CUDNN_CROSS_CORRELATION));

	}

	//初始化输出数据配置
	void initializeOutputDataTensorDescriptor() {

		//获取预期输出数据配置
		checkCUDNN(
				cudnnGetConvolution2dForwardOutputDim(
						this->convolutionDescriptor,
						this->inputDataTensorDescriptor,
						this->kernelTensorDescriptor, &this->outputBatchSize,
						&this->outputFeaturemaps, &this->outputFeaturemapHeight,
						&this->outputFeaturemapWidth));

		checkCUDNN(
				cudnnCreateTensorDescriptor(&this->outputDataTensorDescriptor));
		checkCUDNN(
				cudnnSetTensor4dDescriptor(this->outputDataTensorDescriptor,
						CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
						this->outputBatchSize, this->outputFeaturemaps,
						this->outputFeaturemapHeight,
						this->outputFeaturemapWidth));
	}

public:

	//初始化
	CuConvolutionLayer(cudnnHandle_t cudnnHandle, int inputFeaturemaps,
			int inputBatchSize, int inputFeaturemapHeight,
			int inputFeaturemapWidth, int kernelHeight, int kernelWidth,
			int outputFeaturemaps) {

		this->cudnnHandle = cudnnHandle;

		//输入数据初始化
		this->inputFeaturemaps = inputFeaturemaps;
		this->inputBatchSize = inputBatchSize;
		this->inputFeaturemapHeight = inputFeaturemapHeight;
		this->inputFeaturemapWidth = inputFeaturemapWidth;
		this->initializeInputDataTensorDescriptor();

		//卷积核初始化
		this->kernelHeight = kernelHeight;
		this->kernelWidth = kernelWidth;
		this->outputFeaturemaps = outputFeaturemaps;
		this->initializeKernelTensorDescriptor();

		//卷积操作初始化
		//零填充的行数与列数：0
		this->zeroPaddingHeight = 0, this->zeroPaddingWidth = 0;
		//卷积的水平和垂直的滑动长度：1
		this->verticalFilterStride = 1, this->horizontalFilterStride = 1;
		//横、纵向上取样的比例尺：1
		this->upscaleInXDirection = 1, this->upscaleInYDirection = 1;
		this->initializeConvolutionDescriptor();

		//输出数据初始化
		this->initializeOutputDataTensorDescriptor();

	}

	//卷积运算
	float * convolution(float * u_input_data, float * u_kernel) {

		int outputDataLength = this->outputBatchSize * this->outputFeaturemaps
				* this->outputFeaturemapHeight * this->outputFeaturemapWidth;

		float * d_output_data;
		checkCudaErrors(
				cudaMalloc((void** ) &d_output_data,
						outputDataLength * sizeof(float)));

		//选择FP算法
		checkCUDNN(
				cudnnGetConvolutionForwardAlgorithm(this->cudnnHandle,
						this->inputDataTensorDescriptor,
						this->kernelTensorDescriptor,
						this->convolutionDescriptor,
						this->outputDataTensorDescriptor,
						CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
						this->workspaceSizeInByte,
						&this->convolutionForwardAlgorithm));

		//设置workspace的大小
		this->workspaceSizeInByte = 0;
		checkCUDNN(
				cudnnGetConvolutionForwardWorkspaceSize(this->cudnnHandle,
						this->inputDataTensorDescriptor,
						this->kernelTensorDescriptor,
						this->convolutionDescriptor,
						this->outputDataTensorDescriptor,
						this->convolutionForwardAlgorithm,
						&this->workspaceSizeInByte));

		//为工作区域分配空间
		checkCudaErrors(
				cudaMalloc(&this->d_cudnn_workspace,
						this->workspaceSizeInByte));

		checkCudaErrors(cudaDeviceSynchronize());

		//卷积运算
		float alpha = 1.0f, beta = 0.0f;
		checkCUDNN(
				cudnnConvolutionForward(this->cudnnHandle, &alpha,
						this->inputDataTensorDescriptor, u_input_data,
						this->kernelTensorDescriptor, u_kernel,
						this->convolutionDescriptor,
						this->convolutionForwardAlgorithm,
						this->d_cudnn_workspace, this->workspaceSizeInByte,
						&beta, this->outputDataTensorDescriptor,
						d_output_data));

		//非统一地址
		float * h_output_data = new float[outputDataLength];
		checkCudaErrors(
				cudaMemcpy(h_output_data, d_output_data,
						sizeof(float) * outputDataLength,
						cudaMemcpyDeviceToHost));
		float * u_output_data = Utility::AllocUnifiedMemory(h_output_data,
				outputDataLength, true);

		for (int i = 0; i < outputDataLength; i++) {
			std::cout << i << " : " << u_output_data[i] << std::endl;
		}

		return u_output_data;

	}

};

#endif /* CUCONVOLUTIONLAYER_H_ */
