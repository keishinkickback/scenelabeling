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

#include "CuNeuralNetwork.h"

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

class ImageProcessor {

public:
	void readRGBImage(char *imagePath, std::vector<float> *redChannel,
			std::vector<float> *greenChannel, std::vector<float> *blueChannel) {

		FreeImage_Initialise(TRUE);

		FIBITMAP* fib;
		fib = FreeImage_Load(FIF_PNG, imagePath, PNG_DEFAULT);
		int width = FreeImage_GetWidth(fib);
		int height = FreeImage_GetHeight(fib);

		RGBQUAD color;

		for (int x = 0; x < width; x++) {

			for (int y = 0; y < height; y++) {

				FreeImage_GetPixelColor(fib, x, y, &color);

				float blue = color.rgbBlue;
				float green = color.rgbGreen;
				float red = color.rgbRed;
				redChannel->push_back(red);
				greenChannel->push_back(green);
				blueChannel->push_back(blue);

			}

		}

		FreeImage_Unload(fib);
		FreeImage_DeInitialise();
	}

	std::vector<float> imageChannelNormalization(std::vector<float> *channel) {

		float maxColorChannel = *std::max_element(channel->begin(),
				channel->end());
		float minColorChannel = *std::min_element(channel->begin(),
				channel->end());

		std::vector<float> result;

		for (int i = 0; i < channel->size(); i++) {
			result.push_back(
					(channel->at(i) - minColorChannel)
							/ (maxColorChannel - minColorChannel));
		}

		channel->clear();

		return result;

	}
};

class KernelGenerator {

public:

	void initializeKernelUsingXavierAlgorithm(int kernelHeight,
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

//		for(int i = 0 ; i < kernel->size();i++){
//			std::cout << kernel->at(i) << std::endl;
//		}

	}

};

//class CuNeuralNetwork {
//
//public:
//
//	//input
//	cudnnHandle_t *cudnnHandle;
//	cudnnTensorDescriptor_t * inputDataTensorDescriptor;
//	cudnnFilterDescriptor_t * kernelDescriptor;
//	cudnnConvolutionDescriptor_t * convolutionDescriptor;
//	cudnnTensorDescriptor_t * outputDataTensorDescriptor;
//	int executeBatchSize;
//	int imageHeight;
//	int imageWidth;
//	int kernelHeight;
//	int kernelWidth;
//	int inputChannels;
//	int outputChannels;
//
//	//output
//	size_t workspaceSizeInByte;
//	cudnnConvolutionFwdAlgo_t * algorithm;
//	int outputImageNumber;
//	int outputChannelsOfEachImage;
//	int outputFeaturemapHeight;
//	int outputFeaturemapWidth;
//
//public:
//
//	size_t initializeConvolutionalLayerTensorDescriptor(
//			cudnnHandle_t *cudnnHandle,
//			cudnnTensorDescriptor_t * inputDataTensorDescriptor,
//			cudnnFilterDescriptor_t * kernelDescriptor,
//			cudnnConvolutionDescriptor_t * convolutionDescriptor,
//			cudnnTensorDescriptor_t * outputDataTensorDescriptor,
//			cudnnConvolutionFwdAlgo_t * algorithm, int executeBatchSize,
//			int imageHeight, int imageWidth, int kernelHeight, int kernelWidth,
//			int inputChannels, int outputChannels) {
//
//		//输入数据设定
//		checkCUDNN(cudnnCreateTensorDescriptor(inputDataTensorDescriptor));
//		checkCUDNN(
//				cudnnSetTensor4dDescriptor(*inputDataTensorDescriptor,
//						CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, executeBatchSize,
//						inputChannels, imageHeight, imageWidth));
//
//		//卷积核设定
//		checkCUDNN(cudnnCreateFilterDescriptor(kernelDescriptor));
//		checkCUDNN(
//				cudnnSetFilter4dDescriptor(*kernelDescriptor, CUDNN_DATA_FLOAT,
//						outputChannels, outputChannels, kernelHeight,
//						kernelWidth));
//
//		//卷积操作设定
//		checkCUDNN(cudnnCreateConvolutionDescriptor(convolutionDescriptor));
//		//零填充的行数与列数：0 卷积的水平和垂直的滑动长度：1 x，y向上取样的比例尺：1
//		//不使用卷积操作，因为卷积操作要旋转卷积核，而互相关操作无需旋转卷积核
//		checkCUDNN(
//				cudnnSetConvolution2dDescriptor(*convolutionDescriptor, 0, 0, 1,
//						1, 1, 1, CUDNN_CROSS_CORRELATION));
//
//		//输出数据设定
//		//获取：图片数量，输出featuremap数量，featuremap的高度，featuremap的宽度
//
//		checkCUDNN(
//				cudnnGetConvolution2dForwardOutputDim(*convolutionDescriptor,
//						*inputDataTensorDescriptor, *kernelDescriptor,
//						&this->imageNumber, &this->ChannelsOfImage,
//						&this->featuremapHeight, &this->featuremapWidth));
//
//		checkCUDNN(cudnnCreateTensorDescriptor(outputDataTensorDescriptor));
//		checkCUDNN(
//				cudnnSetTensor4dDescriptor(*outputDataTensorDescriptor,
//						CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
//						outputData->imageNumber, outputData->ChannelsOfImage,
//						outputData->featuremapHeight,
//						outputData->featuremapWidth));
//
//		//选择FP算法
//		checkCUDNN(
//				cudnnGetConvolutionForwardAlgorithm(*cudnnHandle,
//						*inputDataTensorDescriptor, *kernelDescriptor,
//						*convolutionDescriptor, *outputDataTensorDescriptor,
//						CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algorithm));
//
//		//获取workspace的大小
//		size_t workspaceSizeInByte = 0;
//		checkCUDNN(
//				cudnnGetConvolutionForwardWorkspaceSize(*cudnnHandle,
//						*inputDataTensorDescriptor, *kernelDescriptor,
//						*convolutionDescriptor, *outputDataTensorDescriptor,
//						*algorithm, &workspaceSizeInByte));
//
//		return workspaceSizeInByte;
//
//	}
//
//};

class Utility {

public:
	static float* VectorToArray(std::vector<float> * input) {

		float * array;
		array = new float[input->size()];

		for (int i = 0; i < input->size(); i++) {
			array[i] = input->at(i);
		}

		return array;
	}
};

class TestCase {

public:
	static void TestCase1(float * data, float * kernel) {

		float sum = 0.0f;

		sum += data[0] * kernel[0];
		sum += data[1] * kernel[1];
		sum += data[2] * kernel[2];
		sum += data[540] * kernel[3];
		sum += data[541] * kernel[4];
		sum += data[542] * kernel[5];
		sum += data[1080] * kernel[6];
		sum += data[1081] * kernel[7];
		sum += data[1082] * kernel[8];

		std::cout << " Ground Truth : " << sum << std::endl;

	}

	static void printDynamicArray(float * array, int length) {
		for (int i = 0; i < length; i++) {
			std::cout << array[i] << std::endl;
		}
	}

};

int main() {

	int filePathMaxLength = 256;
	int imageHeight = 540;
	int imageWidth = 540;
	int kernelHeight = 3;
	int kernelWidth = 3;

	//读取图片到RGB三个通道
	char imagePath[filePathMaxLength];
	getcwd(imagePath, filePathMaxLength);
	strcat(imagePath, "/trainingset/");
	strcat(imagePath, "train1.png");

	std::vector<float> redChannel;
	std::vector<float> greenChannel;
	std::vector<float> blueChannel;

	ImageProcessor processor;
	processor.readRGBImage(imagePath, &redChannel, &greenChannel, &blueChannel);

	redChannel = processor.imageChannelNormalization(&redChannel);
	greenChannel = processor.imageChannelNormalization(&greenChannel);
	blueChannel = processor.imageChannelNormalization(&blueChannel);

	float * h_data;
	h_data = Utility::VectorToArray(&redChannel);
	//END

	//卷积核数据初始化
	KernelGenerator generator;
	std::vector<float> kernel(kernelHeight * kernelWidth);
	generator.initializeKernelUsingXavierAlgorithm(kernelHeight, kernelWidth, 1,
			&kernel);

	float * h_kernel;
	h_kernel = Utility::VectorToArray(&kernel);
	//END

	//GPU的查询与选择
	int GPUs;
	checkCudaErrors(cudaGetDeviceCount(&GPUs));
	if (GPUs > 0) {
		checkCudaErrors(cudaSetDevice(0));
	} else {
		return 0;
	}

	//cudnn初始化
	cudnnHandle_t cudnnHandle = NULL;
	cudnnCreate(&cudnnHandle);

	cudnnTensorDescriptor_t inputDataTensor;
	cudnnFilterDescriptor_t kernelDescriptor;
	cudnnConvolutionDescriptor_t convolutionDescriptor;
	cudnnTensorDescriptor_t outputDataTensor;
	cudnnConvolutionFwdAlgo_t algorithm;
	int executeBatchSize = 1;
	int inputFeaturemaps = 1;
	int outputFeaturemaps = 1;
	size_t workspaceSizeInByte;
	int outputImages;
	int outputFeaturemapsForEachImage;
	int outputFeaturemapHeight;
	int outputFeaturemapWidth;

	CuNeuralNetwork network;

	network.initializeConvolutionalLayerTensorDescriptor(&cudnnHandle,
			&inputDataTensor, &kernelDescriptor, &convolutionDescriptor,
			&outputDataTensor, &algorithm, executeBatchSize, imageWidth,
			imageHeight, kernelHeight, kernelWidth, inputFeaturemaps,
			outputFeaturemaps, &workspaceSizeInByte, &outputImages,
			&outputFeaturemapsForEachImage, &outputFeaturemapHeight,
			&outputFeaturemapWidth);

	std::cout << outputImages << " " << outputFeaturemapsForEachImage << " "
			<< outputFeaturemapHeight << " " << outputFeaturemapWidth
			<< std::endl;

	//输入数据设定
//	cudnnTensorDescriptor_t inputDataTensor;
//	checkCUDNN(cudnnCreateTensorDescriptor(&inputDataTensor));
//	//第一层输入为图片原始数据，看做第一层的featuremap,数量为1
//	checkCUDNN(
//			cudnnSetTensor4dDescriptor(inputDataTensor, CUDNN_TENSOR_NCHW,
//					CUDNN_DATA_FLOAT, 1, 1, imageHeight, imageWidth));

//device上分配内存空间
	float *d_data;
	checkCudaErrors(
			cudaMalloc(&d_data,
					sizeof(float) * 1 * 1 * imageHeight * imageWidth));

	//拷贝数据到device
	checkCudaErrors(
			cudaMemcpyAsync(d_data, h_data,
					sizeof(float) * 1 * 1 * imageHeight * imageWidth,
					cudaMemcpyHostToDevice));

	//卷积核设定
	//输入featuremap为1个，即图片原始数据。输出为1个featuremap，所以需要1个卷积核。
//	int inputFeaturemapNumber = 1;
//	int outputFeaturemapNumber = 1;
//	cudnnFilterDescriptor_t kernelDescriptor;
//	checkCUDNN(cudnnCreateFilterDescriptor(&kernelDescriptor));
//	checkCUDNN(
//			cudnnSetFilter4dDescriptor(kernelDescriptor, CUDNN_DATA_FLOAT,
//					outputFeaturemapNumber, inputFeaturemapNumber, kernelHeight,
//					kernelWidth));
	//device上分配内存空间
	float *d_kernel;
	checkCudaErrors(
			cudaMalloc(&d_kernel,
					sizeof(float) * 1 * 1 * kernelHeight * kernelWidth));
	//拷贝数据到device
	checkCudaErrors(
			cudaMemcpyAsync(d_kernel, h_kernel,
					sizeof(float) * 1 * 1 * kernelHeight * kernelWidth,
					cudaMemcpyHostToDevice));

//	//卷积操作设定
//	cudnnConvolutionDescriptor_t convolutionDescriptor;
//	checkCUDNN(cudnnCreateConvolutionDescriptor(&convolutionDescriptor));
//	//零填充的行数与列数：0,卷积的水平和垂直的滑动长度：1,x，y向上取样的比例尺：1
//	//不使用卷积操作，因为卷积操作要旋转卷积核，而这里不需要旋转，互相关就是无需旋转的卷积乘法
//	checkCUDNN(
//			cudnnSetConvolution2dDescriptor(convolutionDescriptor, 0, 0, 1, 1,
//					1, 1, CUDNN_CROSS_CORRELATION));
//
//	//获取：图片数量，输出featuremap数量，featuremap的高度，featuremap的宽度
//	int outputN, outputC, outputH, outputW;
//	checkCUDNN(
//			cudnnGetConvolution2dForwardOutputDim(convolutionDescriptor,
//					inputDataTensor, kernelDescriptor, &outputN, &outputC,
//					&outputH, &outputW));
//
//	//输出featuremap设定
//	cudnnTensorDescriptor_t outputDataTensor;
//	checkCUDNN(cudnnCreateTensorDescriptor(&outputDataTensor));
//	checkCUDNN(
//			cudnnSetTensor4dDescriptor(outputDataTensor, CUDNN_TENSOR_NCHW,
//					CUDNN_DATA_FLOAT, outputN, outputC, outputH, outputW));
	//device上分配内存空间
	float *d_output_data;
//	checkCudaErrors(
//			cudaMalloc(&d_output_data,
//					sizeof(float) * adapter.OutputData.imageNumber
//							* adapter.OutputData.ChannelsOfImage
//							* adapter.OutputData.featuremapHeight
//							* adapter.OutputData.featuremapWidth));

//	//选择fp算法
//	cudnnConvolutionFwdAlgo_t algorithm;
//	checkCUDNN(
//			cudnnGetConvolutionForwardAlgorithm(cudnnHandle, inputDataTensor,
//					kernelDescriptor, convolutionDescriptor, outputDataTensor,
//					CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algorithm));
//
//	//获取workspace的大小
//	size_t workspaceSizeInByte = 0;
//	checkCUDNN(
//			cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
//					inputDataTensor, kernelDescriptor, convolutionDescriptor,
//					outputDataTensor, algorithm, &workspaceSizeInByte));
	//device上分配内存空间
//void *d_cudnn_workspace = nullptr;
//																	checkCudaErrors(cudaMalloc(&d_cudnn_workspace, workspaceSizeInByte));
//
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	//fp
//	float alpha = 1.0f, beta = 0.0f;
//
//	checkCUDNN(
//			cudnnConvolutionForward(cudnnHandle, &alpha, inputDataTensor,
//					d_data, kernelDescriptor, d_kernel, convolutionDescriptor,
//					algorithm, d_cudnn_workspace, workspaceSizeInByte, &beta,
//					outputDataTensor, d_output_data));
//
//	//定义动态float数组
//	float * h_output_data = new float[outputN * outputC * outputH * outputH];
//
//	checkCudaErrors(
//			cudaMemcpyAsync(h_output_data, d_output_data,
//					sizeof(float) * outputN * outputC * outputH * outputH,
//					cudaMemcpyDeviceToHost));
//
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	TestCase::TestCase1(h_data, h_kernel);
//
//	std::cout << " test result : " << h_output_data[0] << std::endl;

	//checkCUDNN(cudnnDestroyTensorDescriptor(redChannelDataTensor));

}

