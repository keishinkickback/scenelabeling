/*
 ============================================================================
 Name        : scenelabeling.cu
 Author      : liujinhang @ whut
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

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

#include "includes/CuConvolutionLayer.h"
#include "includes/CuNeuralNetwork.h"
#include "includes/ImageProcessor.h"
#include "includes/KernelGenerator.h"
#include "includes/Utility.h"
#include "includes/TestCase.h"

#include <vector>

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

class SceneLabeling {
public:
	float * convolution_maxpooling_sigmoid(cudnnHandle_t cudnnHandle,
			float * h_input_data, int batchSize, int imageHeight,
			int imageWidth, int kernelHeight, int kernelWidth,
			int poolingWindowHeight, int poolingWindowWidth,
			int inputFeaturemaps, int outputFeaturemaps) {

		//卷积核初始化
		float * h_kernel;
		std::vector<float> kernels;
		for (int i = 0; i < inputFeaturemaps * outputFeaturemaps; i++) {
			std::vector<float> kernel(kernelHeight * kernelWidth);
			KernelGenerator::InitializeKernelUsingXavierAlgorithm(kernelHeight,
					kernelWidth, inputFeaturemaps, &kernel);
			kernels.insert(kernels.end(), kernel.begin(), kernel.end());
		}
		h_kernel = Utility::VectorToDynamicArray(&kernels);

		//TEST
		std::cout << " size of kernel : " << kernels.size()
				<< " with output featuremaps : "
				<< inputFeaturemaps * outputFeaturemaps << std::endl;

		//偏置项初始化
		//与卷积核生成采用同样的方法同样的数量
		float * h_bias;
		std::vector<float> bias(outputFeaturemaps);
		KernelGenerator::InitializeBiasUsingXavierAlgorithm(outputFeaturemaps,
				&bias);
		h_bias = Utility::VectorToDynamicArray(&bias);

		//TEST
		std::cout << " size of bias : " << bias.size()
				<< " with output featuremaps : " << outputFeaturemaps
				<< std::endl;

		//神经网络初始化
		CuNeuralNetwork network;

		//输入数据设定
		cudnnTensorDescriptor_t inputDataTensorDescriptor;
		float * d_data = network.initializeInputDataLayer(h_input_data,
				&inputDataTensorDescriptor, batchSize, inputFeaturemaps,
				imageHeight, imageWidth);

		//卷积核设定
		cudnnFilterDescriptor_t kernelDescriptor;
		float * d_kernel = network.initializeKernels(h_kernel,
				&kernelDescriptor, inputFeaturemaps, outputFeaturemaps,
				kernelHeight, kernelWidth);

		//卷积操作设定
		cudnnConvolutionDescriptor_t convolutionDescriptor;
		checkCUDNN(cudnnCreateConvolutionDescriptor(&convolutionDescriptor));
		//零填充的行数与列数：0
		//卷积的水平和垂直的滑动长度：1
		//横、纵向上取样的比例尺：1
		//不使用卷积操作，因为卷积操作要旋转卷积核，而这里不需要旋转，互相关就是无需旋转的卷积乘法。
		checkCUDNN(
				cudnnSetConvolution2dDescriptor(convolutionDescriptor, 0, 0, 1,
						1, 1, 1, CUDNN_CROSS_CORRELATION));

		//输出数据设定
		cudnnTensorDescriptor_t outputDataTensorDescriptor;
		cudnnOutputDim outputDim;
		float *d_output_data = network.initializeOutputDataLayer(
				&inputDataTensorDescriptor, &kernelDescriptor,
				&convolutionDescriptor, &outputDataTensorDescriptor,
				&outputDim);

		//选择FP算法
		cudnnConvolutionFwdAlgo_t algorithm;
		checkCUDNN(
				cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
						inputDataTensorDescriptor, kernelDescriptor,
						convolutionDescriptor, outputDataTensorDescriptor,
						CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algorithm));

		//设置workspace的大小
		size_t workspaceSizeInByte = 0;
		checkCUDNN(
				cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
						inputDataTensorDescriptor, kernelDescriptor,
						convolutionDescriptor, outputDataTensorDescriptor,
						algorithm, &workspaceSizeInByte));

void *d_cudnn_workspace = nullptr;
										checkCudaErrors(cudaMalloc(&d_cudnn_workspace, workspaceSizeInByte));

		//卷积运算
		float alpha = 1.0f, beta = 0.0f;
		checkCUDNN(
				cudnnConvolutionForward(cudnnHandle, &alpha,
						inputDataTensorDescriptor, d_data, kernelDescriptor,
						d_kernel, convolutionDescriptor, algorithm,
						d_cudnn_workspace, workspaceSizeInByte, &beta,
						outputDataTensorDescriptor, d_output_data));

		checkCudaErrors(cudaDeviceSynchronize());

		//偏置项设定
		cudnnTensorDescriptor_t biasTensorDescriptor;
		float *d_bias = network.addBiasUnits(h_bias, &biasTensorDescriptor,
				outputFeaturemaps, kernelHeight, kernelWidth);

		//加上偏置项
		alpha = 1.0f, beta = 1.0f;
		checkCUDNN(
				cudnnAddTensor(cudnnHandle, CUDNN_ADD_SAME_C, &alpha,
						biasTensorDescriptor, d_bias, &beta,
						outputDataTensorDescriptor, d_output_data));

		checkCudaErrors(cudaDeviceSynchronize());

		//池化设定
		alpha = 1.0f, beta = 0.0f;
		cudnnPoolingDescriptor_t poolingDescriptor;
		cudnnTensorDescriptor_t poolingDataTensorDescriptor;
		cudnnOutputDim poolingOutputDim;
		int poolingVerticalStride = 1;
		int poolingHorizontalStride = 1;
		float * d_pooling_output_data = network.initializePoolingLayer(
				d_output_data, &outputDataTensorDescriptor, &poolingDescriptor,
				&poolingDataTensorDescriptor, &outputDim, poolingWindowHeight,
				poolingWindowWidth, poolingVerticalStride,
				poolingHorizontalStride, &poolingOutputDim);

		//池化运算
		checkCUDNN(
				cudnnPoolingForward(cudnnHandle, poolingDescriptor, &alpha,
						outputDataTensorDescriptor, d_output_data, &beta,
						poolingDataTensorDescriptor, d_pooling_output_data));

		checkCudaErrors(cudaDeviceSynchronize());

		//激活函数设定
		alpha = 1.0f, beta = 0.0f;
		float * d_activation_output_data;

		checkCudaErrors(
				cudaMalloc(&d_activation_output_data,
						sizeof(float) * poolingOutputDim.outputImages
								* poolingOutputDim.outputFeaturemapsForEachImage
								* poolingOutputDim.outputFeaturemapHeight
								* poolingOutputDim.outputFeaturemapWidth));

		checkCUDNN(
				cudnnActivationForward(cudnnHandle, CUDNN_ACTIVATION_SIGMOID,
						&alpha, poolingDataTensorDescriptor,
						d_pooling_output_data, &beta,
						poolingDataTensorDescriptor, d_activation_output_data));

		checkCudaErrors(cudaDeviceSynchronize());

		//Test section
		//d_output_data数据回传
		float * h_output_data = new float[outputDim.outputImages
				* outputDim.outputFeaturemapsForEachImage
				* outputDim.outputFeaturemapHeight
				* outputDim.outputFeaturemapWidth];
		checkCudaErrors(
				cudaMemcpyAsync(h_output_data, d_output_data,
						sizeof(float) * outputDim.outputImages
								* outputDim.outputFeaturemapsForEachImage
								* outputDim.outputFeaturemapHeight
								* outputDim.outputFeaturemapWidth,
						cudaMemcpyDeviceToHost));

		//d_pooling_output_data数据回传
		float * h_pooling_output_data = new float[poolingOutputDim.outputImages
				* poolingOutputDim.outputFeaturemapsForEachImage
				* poolingOutputDim.outputFeaturemapHeight
				* poolingOutputDim.outputFeaturemapWidth];
		checkCudaErrors(
				cudaMemcpyAsync(h_pooling_output_data, d_pooling_output_data,
						sizeof(float) * poolingOutputDim.outputImages
								* poolingOutputDim.outputFeaturemapsForEachImage
								* poolingOutputDim.outputFeaturemapHeight
								* poolingOutputDim.outputFeaturemapWidth,
						cudaMemcpyDeviceToHost));

		//d_activation_output_data数据回传
		float * h_activation_output_data =
				new float[poolingOutputDim.outputImages
						* poolingOutputDim.outputFeaturemapsForEachImage
						* poolingOutputDim.outputFeaturemapHeight
						* poolingOutputDim.outputFeaturemapWidth];
		checkCudaErrors(
				cudaMemcpyAsync(h_activation_output_data,
						d_activation_output_data,
						sizeof(float) * poolingOutputDim.outputImages
								* poolingOutputDim.outputFeaturemapsForEachImage
								* poolingOutputDim.outputFeaturemapHeight
								* poolingOutputDim.outputFeaturemapWidth,
						cudaMemcpyDeviceToHost));

		//测试用例：卷积运算测试（不特定featuremap单batchSize测试）
		TestCase::CPUConvolutionOperation(h_input_data, h_kernel, h_bias,
				h_output_data, imageHeight, imageWidth, kernelHeight,
				kernelWidth, inputFeaturemaps, outputFeaturemaps, outputDim);

		//测试用例：max pooling测试
		TestCase::SingleCPUMaxPoolingOperation(h_output_data,
				h_pooling_output_data, outputDim.outputFeaturemapHeight,
				outputDim.outputFeaturemapWidth, poolingWindowHeight,
				poolingWindowWidth);

		//Destroy section
		checkCUDNN(cudnnDestroyTensorDescriptor(inputDataTensorDescriptor));
		checkCUDNN(cudnnDestroyTensorDescriptor(outputDataTensorDescriptor));
		checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDescriptor));
		checkCUDNN(cudnnDestroyTensorDescriptor(poolingDataTensorDescriptor));
		checkCUDNN(cudnnDestroyFilterDescriptor(kernelDescriptor));
		checkCUDNN(cudnnDestroyConvolutionDescriptor(convolutionDescriptor));
		checkCUDNN(cudnnDestroyPoolingDescriptor(poolingDescriptor));

		return h_activation_output_data;
	}

	void computeFullconnectedDataLayer(cublasHandle_t cublasHandle,
			float * d_input_data, float * d_kernel, float * d_output_data,
			float * d_bias, float * d_ones_vector, int outputFeaturemaps,
			int inputFeaturemaps, int inputFeaturemapHeight,
			int inputFeaturemapWidth) {

		int featuremapsLength = inputFeaturemapHeight * inputFeaturemapWidth
				* inputFeaturemaps;

//		cudnnTensorDescriptor_t fullConnectedTensorDescriptor;
//		checkCUDNN(cudnnCreateTensorDescriptor(&fullConnectedTensorDescriptor));
//
//		checkCUDNN(
//				cudnnSetTensor4dDescriptor(fullConnectedTensorDescriptor,
//						CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize,
//						inputFeaturemaps, 1, 1));

//		//grid长宽设定，一次运算只能有一个grid参与，所以第三项恒为1
//		dim3 gridParameter(10, 10, 1);
//		//block长宽高设定
//		dim3 blockParameter(10, 10, 10);
//		float *d_ones_vector;
//		checkCudaErrors(cudaMalloc(&d_ones_vector, sizeof(float) * inputFeaturemaps));
//		FillOnes<<<gridParameter, blockParameter>>>(d_ones_vector, inputFeaturemaps);

//		float * d_ones_vector = this->initializeOnesVector(1);

		float alpha = 1.0;
		float beta = 0;

		//全连接运算
		checkCudaErrors(
				cublasSgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, outputFeaturemaps, 1, featuremapsLength, &alpha, d_input_data, featuremapsLength, d_kernel, featuremapsLength, &beta, d_output_data, outputFeaturemaps));

		//加上偏置项
		//d_ones_vector应为1*1
		beta = 1;
		checkCudaErrors(
				cublasSgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, outputFeaturemaps, 1, 1, &alpha, d_bias, featuremapsLength, d_ones_vector, featuremapsLength, &beta, d_output_data, outputFeaturemaps));

	}

};

int main(void) {

	//读取图片
	char imagePath[256];
	getcwd(imagePath, 256);
	//训练集文件夹相对地址
	strcat(imagePath, "/trainingset/");
	//训练样本1
	strcat(imagePath, "train1.png");

	//初始化三通道向量容器
	std::vector<float> redChannel;
	std::vector<float> greenChannel;
	std::vector<float> blueChannel;
	std::vector<float> input_data;

	//从图片里读取三通道数据
	ImageProcessor::ReadRGBImage(imagePath, &redChannel, &greenChannel,
			&blueChannel);
	//三通道数据分别归一化
	Utility::MinMaxNormalization(redChannel);
	Utility::MinMaxNormalization(greenChannel);
	Utility::MinMaxNormalization(blueChannel);
	//合并三通道数据到一个向量
	input_data.insert(input_data.end(), redChannel.begin(), redChannel.end());
	input_data.insert(input_data.end(), greenChannel.begin(),
			greenChannel.end());
	input_data.insert(input_data.end(), blueChannel.begin(), blueChannel.end());
	//转换为动态数组
	float * h_input_data = Utility::VectorToDynamicArray(&input_data);
	//统一寻址转换
	float * u_input_data = Utility::AllocUnifiedMemory(h_input_data,
			input_data.size(), true);

	//GPU查找与设定
	int GPUs;
	checkCudaErrors(cudaGetDeviceCount(&GPUs));
	std::cout << " GPUs : " << GPUs << std::endl;
	if (GPUs > 0) {
		checkCudaErrors(cudaSetDevice(0));
	} else {
		return 0;
	}

	//cudnn初始化
	cudnnHandle_t cudnnHandle = NULL;
	cudnnCreate(&cudnnHandle);

	//设定第一层卷积运算参数
	int inputFeaturemaps = 3;
	int inputBatchSize = 1;
	int inputFeaturemapHeight = 540;
	int inputFeaturemapWidth = 540;
	int kernelHeight = 3;
	int kernelWidth = 3;
	int outputFeaturemaps = 3;

	//初始化卷积核
	std::vector<float> kernel = KernelGenerator::InitializeKernels(
			inputFeaturemaps, outputFeaturemaps, kernelHeight, kernelWidth);
	float * h_kernel = Utility::VectorToDynamicArray(&kernel);
	//统一寻址转换
	float * u_kernel = Utility::AllocUnifiedMemory(h_kernel, kernel.size(),
			true);

	//初始化卷基层的配置
	CuConvolutionLayer fristConvolutionLayer(cudnnHandle, inputFeaturemaps,
			inputBatchSize, inputFeaturemapHeight, inputFeaturemapWidth,
			kernelHeight, kernelWidth, outputFeaturemaps);

	//卷积运算
	float * u_frist_convolution_layer_output_data =
			fristConvolutionLayer.convolution(u_input_data, u_kernel);



//	SceneLabeling sceneLabeling;
//
//	float * first_group_output_data =
//			sceneLabeling.convolution_maxpooling_sigmoid(cudnnHandle,
//					h_input_data, batchSize, imageHeight, imageWidth,
//					kernelHeight, kernelWidth, poolingWindowHeight,
//					poolingWindowWidth, inputFeaturemaps,
//					outputFeaturemaps);

//学习速率计算

//		float initialLearningRate = 1.0;
//		float learningRateGamma = 0.1;
//		float learningRatePower = 2;
//
//		int iter = 1;
//
//		float learningRate = static_cast<float>(initialLearningRate
//				* pow((1.0 + initialLearningRate * iter), (-learningRatePower)));
//
//		std::cout << " learning rate : " << learningRate << std::endl;
//
//		//softmax损失函数
//		checkCUDNN(
//				cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE,
//						CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, fc2Tensor, fc2,
//						&beta, fc2Tensor, result));

	return 1;

}

// 填充全一向量核函数
__global__ void FillOnes(float *vector, int length) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= length)
		return;

	vector[idx] = 1.0f;
}

__global__ void SoftmaxLossBackprop(const float *label, int num_labels,
		int batch_size, float *diff) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batch_size)
		return;

	const int label_value = static_cast<int>(label[idx]);

	// For each item in the batch, decrease the result of the label's value by 1
	diff[idx * num_labels + label_value] -= 1.0f;
}

//全连接矩阵测试
//int main(void) {
//
//	int inputFeaturemapHeight = 2;
//	int inputFeaturemapWidth = 3;
//	int inputFeaturemaps = 2;
//	int batchSize = 2;
//
//	//grid长宽设定，一次运算只能有一个grid参与，所以第三项恒为1
//	dim3 gridParameter(10, 10, 1);
//	//block长宽高设定
//	dim3 blockParameter(10, 10, 10);
//	float *d_ones_vector;
//	checkCudaErrors(
//			cudaMalloc(&d_ones_vector, sizeof(float) * inputFeaturemaps));
//	FillOnes<<<gridParameter, blockParameter>>>(d_ones_vector,
//			inputFeaturemaps);
//
//	int featuremapVectorLength = inputFeaturemapHeight * inputFeaturemapWidth
//			* inputFeaturemaps * batchSize;
//	int kernelVectorLength = inputFeaturemapHeight * inputFeaturemapWidth
//			* inputFeaturemaps;
//	int outputFeaturemapVectorLength = batchSize;
//
//	float h_input_featuremaps[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
//			14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
//	float h_kernel[] = { 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };
//	float h_output_featuremaps[outputFeaturemapVectorLength];
//	float h_bias[] = { 100, 200 };
//	float *d_a, *d_b, *d_c, *d_d;
//	checkCudaErrors(
//			cudaMalloc((void** ) &d_a, featuremapVectorLength * sizeof(float)));
//	checkCudaErrors(
//			cudaMalloc((void** ) &d_b, kernelVectorLength * sizeof(float)));
//	checkCudaErrors(
//			cudaMalloc((void** ) &d_c,
//					outputFeaturemapVectorLength * sizeof(float)));
//	checkCudaErrors(
//			cudaMalloc((void** ) &d_d,
//					outputFeaturemapVectorLength * sizeof(float)));
//	checkCudaErrors(
//			cudaMemcpy(d_a, &h_input_featuremaps,
//					featuremapVectorLength * sizeof(float),
//					cudaMemcpyHostToDevice));
//	checkCudaErrors(
//			cudaMemcpy(d_b, &h_kernel, kernelVectorLength * sizeof(float),
//					cudaMemcpyHostToDevice));
//	checkCudaErrors(
//			cudaMemset(d_c, 0, outputFeaturemapVectorLength * sizeof(float)));
//	checkCudaErrors(
//			cudaMemcpy(d_d, &h_bias,
//					outputFeaturemapVectorLength * sizeof(float),
//					cudaMemcpyHostToDevice));
//	cublasHandle_t handle;
//	cublasCreate(&handle);
//
//	SceneLabeling sceneLabeling;
//
//	sceneLabeling.computeFullconnectedDataLayer(handle, d_a, d_b, d_c, d_d,
//			d_ones_vector, batchSize, inputFeaturemaps, inputFeaturemapHeight,
//			inputFeaturemapWidth);
//
//	checkCudaErrors(
//			cudaMemcpy(h_output_featuremaps, d_c,
//					outputFeaturemapVectorLength * sizeof(float),
//					cudaMemcpyDeviceToHost));
//	for (int i = 0; i < outputFeaturemapVectorLength; i++) {
//		printf("%f\n", h_output_featuremaps[i]);
//	}
//	printf("\n");
//	return 0;
//}
